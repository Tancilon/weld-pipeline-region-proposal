from __future__ import annotations

import torch

from datasets.datasets_omni6dpose import Omni6DPoseDataSet, get_data_loaders_from_cfg


class FrontendROIMaskDataset(Omni6DPoseDataSet):
    def __getitem__(self, index):
        sample = super().__getitem__(index)
        return {
            "roi_rgb": sample["roi_rgb"],
            "roi_rgb_": sample["roi_rgb_"],
            "roi_mask": sample["roi_mask"].to(dtype=torch.float32).contiguous(),
            "class_label": torch.as_tensor(sample["class_label"], dtype=torch.long),
            "object_name": sample["object_name"],
            "class_name": sample["class_name"],
        }


def get_frontend_data_loaders_from_cfg(cfg, data_type=["train", "val", "test"]):
    return get_data_loaders_from_cfg(
        cfg=cfg, data_type=data_type, dataset_cls=FrontendROIMaskDataset
    )
