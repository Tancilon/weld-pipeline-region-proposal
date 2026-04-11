import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from runners import visualize_seg_single_agent as vis_runner
from runners.visualize_seg_single_agent import normalize_image_names, select_sample_indices


def test_normalize_image_names_splits_and_trims_entries():
    assert normalize_image_names("foo.png, bar.png ,, baz.png") == [
        "foo.png",
        "bar.png",
        "baz.png",
    ]


def test_select_sample_indices_uses_image_names_before_num_vis():
    dataset = SimpleNamespace(
        image_ids=[10, 11, 12],
        coco=SimpleNamespace(
            loadImgs=lambda image_id_list: [
                {"file_name": {10: "foo.png", 11: "bar.png", 12: "baz.png"}[image_id_list[0]]}
            ]
        ),
    )

    indices, missing = select_sample_indices(dataset, ["bar.png", "missing.png"])

    assert indices == [1]
    assert missing == ["missing.png"]


def test_select_sample_indices_falls_back_to_num_vis_when_no_image_names():
    dataset = SimpleNamespace(image_ids=[10, 11, 12, 13])

    indices, missing = select_sample_indices(dataset, None, num_vis=2)

    assert indices == [0, 1]
    assert missing == []


def test_select_sample_indices_requires_num_vis_when_image_names_absent():
    dataset = SimpleNamespace(image_ids=[10, 11])

    with pytest.raises(ValueError, match="num_vis"):
        select_sample_indices(dataset, None, num_vis=None)


def test_run_visualization_writes_requested_image_and_handles_no_prediction(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    images_dir = data_dir / "images"
    annotations_dir = data_dir / "annotations"
    output_dir = tmp_path / "out"
    images_dir.mkdir(parents=True)
    annotations_dir.mkdir(parents=True)

    import numpy as np

    image = np.full((32, 32, 3), 200, dtype=np.uint8)
    (images_dir / "foo.png").write_bytes(b"fake-image")
    (annotations_dir / "val.json").write_text("{}", encoding="utf-8")

    fake_cv2 = types.ModuleType("cv2")
    fake_cv2.COLOR_BGR2RGB = 0
    fake_cv2.COLOR_RGB2BGR = 1
    fake_cv2.INTER_LINEAR = 1
    fake_cv2.RETR_EXTERNAL = 0
    fake_cv2.CHAIN_APPROX_SIMPLE = 0
    fake_cv2.imread = lambda path: image.copy()
    fake_cv2.cvtColor = lambda img, code: img
    fake_cv2.resize = lambda img, size, interpolation=None: img
    fake_cv2.addWeighted = lambda src1, alpha, src2, beta, gamma: src1
    fake_cv2.findContours = lambda *args, **kwargs: ([], None)
    fake_cv2.drawContours = lambda *args, **kwargs: None
    fake_cv2.moments = lambda *args, **kwargs: {"m00": 0, "m10": 0, "m01": 0}
    fake_cv2.imwrite = lambda path, vis: Path(path).write_bytes(b"vis") or True
    monkeypatch.setitem(sys.modules, "cv2", fake_cv2)

    class FakeCoco:
        def loadImgs(self, image_id_list):
            return [{"file_name": "foo.png"}]

    class FakeDataset:
        def __init__(self, **kwargs):
            self.image_ids = [101]
            self.coco = FakeCoco()

        def __getitem__(self, index):
            return {
                "roi_rgb": torch.zeros(3, 32, 32),
                "gt_masks": torch.ones(20, 32, 32),
                "gt_classes": torch.tensor([0] + [6] * 19),
                "num_instances": torch.tensor(1),
                "image_id": torch.tensor(101),
            }

    fake_dataset_module = types.ModuleType("datasets.datasets_nuclear")
    fake_dataset_module.CLASS_NAMES = ["盖板", "方管", "喇叭口", "H型钢", "槽钢", "坡口"]
    fake_dataset_module.NuclearWorkpieceDataset = FakeDataset
    fake_dataset_module.collate_nuclear = lambda batch: batch[0]
    fake_dataset_module.process_batch_seg = lambda batch, device: batch
    monkeypatch.setitem(sys.modules, "datasets.datasets_nuclear", fake_dataset_module)

    class FakeNet:
        def eval(self):
            return self

        def __call__(self, batch, mode="segmentation"):
            assert mode == "segmentation"
            class_logits = torch.zeros(1, 1, 7)
            mask_logits = torch.zeros(1, 1, 32, 32)
            return class_logits, mask_logits

    fake_agent = SimpleNamespace(net=FakeNet())
    monkeypatch.setattr(
        vis_runner,
        "_build_agent",
        lambda args: (SimpleNamespace(device="cpu"), fake_agent),
    )

    args = SimpleNamespace(
        nuclear_data_path=str(data_dir),
        seg_ckpt="./results/ckpts/SegNet/best.pth",
        split="val",
        output_dir=str(output_dir),
        num_vis=5,
        score_threshold=0.9,
        image_names="foo.png",
        img_size=32,
        device="cpu",
        num_points=1024,
        repeat_num=10,
    )

    vis_runner.run_visualization(args)

    assert (output_dir / "val_foo.png").exists()


def test_main_strips_sys_argv_before_runtime_imports(monkeypatch):
    original_argv = sys.argv[:]
    seen = {}

    def fake_run_visualization(args):
        seen["argv"] = sys.argv[:]
        seen["args"] = args

    monkeypatch.setattr(vis_runner, "run_visualization", fake_run_visualization)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "visualize_seg_single_agent.py",
            "--nuclear_data_path",
            "./data/aiws5.2-dataset-v1-aug",
            "--seg_ckpt",
            "./results/ckpts/SegNet/best.pth",
            "--split",
            "val",
            "--num_vis",
            "20",
        ],
    )

    try:
        vis_runner.main()
    finally:
        sys.argv = original_argv

    assert seen["argv"] == ["visualize_seg_single_agent.py"]
    assert seen["args"].seg_ckpt == "./results/ckpts/SegNet/best.pth"
