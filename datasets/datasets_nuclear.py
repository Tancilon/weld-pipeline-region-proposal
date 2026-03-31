"""
Dataset loader for nuclear power plant workpiece recognition and segmentation.

Supports 6 workpiece categories and two operating modes:
  - Segmentation training: returns RGB + GT instance masks + GT class labels
  - Full pipeline: additionally returns depth + GT pose + GT size for end-to-end
    segmentation + pose estimation.

Expected annotation format: COCO-style JSON with instance segmentation polygons.
"""

import sys
import os
import cv2
import json
import torch
import numpy as np
import torch.utils.data as data

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from pycocotools.coco import COCO
from pycocotools import mask as coco_mask_utils
from utils.datasets_utils import crop_resize_by_warp_affine, get_2d_coord_np


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CLASS_NAMES = ['盖板', '方管', '喇叭口', 'H型钢', '槽钢', '坡口']
CLASS_TO_IDX = {name: i for i, name in enumerate(CLASS_NAMES)}
NUM_CLASSES = len(CLASS_NAMES)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class NuclearWorkpieceDataset(data.Dataset):
    """Dataset for nuclear workpiece instance segmentation and pose estimation.

    Directory layout (COCO-style)::

        data_dir/
            images/
                000001.png
                000002.png
                ...
            depth/          (optional, for full-pipeline mode)
                000001.exr
                ...
            annotations/
                train.json   (COCO format)
                val.json

    Each annotation entry may optionally carry extra fields for pose estimation
    (``rotation``, ``translation``, ``bbox_side_len``) stored in the annotation
    ``extra`` dict.

    Args:
        cfg: Global config namespace.
        data_dir: Root directory of the dataset.
        annotation_file: Path to the COCO-format annotation JSON.
        mode: ``'seg'`` for segmentation-only training, ``'full'`` for the
            complete pipeline (seg + pose).
        img_size: Crop/resize target size (must be divisible by 14 for DINOv2).
        max_instances: Maximum number of GT instances per image (padded/truncated).
    """

    def __init__(self, cfg, data_dir, annotation_file, mode='seg',
                 img_size=224, max_instances=20):
        assert mode in ('seg', 'full')
        self.cfg = cfg
        self.data_dir = data_dir
        self.mode = mode
        self.img_size = img_size
        self.max_instances = max_instances

        # Load COCO annotations
        self.coco = COCO(annotation_file)
        self.image_ids = sorted(self.coco.getImgIds())

        # Build category id -> our class index mapping
        coco_cats = {cat['id']: cat['name'] for cat in self.coco.loadCats(self.coco.getCatIds())}
        self.catid_to_classidx = {}
        for coco_id, name in coco_cats.items():
            if name in CLASS_TO_IDX:
                self.catid_to_classidx[coco_id] = CLASS_TO_IDX[name]

        print(f"[NuclearWorkpieceDataset] Loaded {len(self.image_ids)} images, "
              f"{len(self.catid_to_classidx)} classes mapped, mode={mode}")

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, index):
        img_id = self.image_ids[index]
        img_info = self.coco.loadImgs(img_id)[0]
        img_path = os.path.join(self.data_dir, 'images', img_info['file_name'])

        # Load RGB
        rgb = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if rgb is None:
            raise FileNotFoundError(f"Image not found: {img_path}")
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        im_H, im_W = rgb.shape[:2]

        # Load annotations for this image
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)

        # Filter to known categories
        anns = [a for a in anns if a['category_id'] in self.catid_to_classidx]

        # Decode masks and class labels
        gt_masks = []
        gt_classes = []
        ann_extras = []  # for pose data if available
        for ann in anns:
            rle = self.coco.annToRLE(ann)
            mask = coco_mask_utils.decode(rle)  # [H, W] uint8
            gt_masks.append(mask)
            gt_classes.append(self.catid_to_classidx[ann['category_id']])
            ann_extras.append(ann.get('extra', {}))

        # Resize RGB to img_size
        rgb_resized = cv2.resize(rgb, (self.img_size, self.img_size),
                                 interpolation=cv2.INTER_LINEAR)
        roi_rgb = self._rgb_transform(rgb_resized)  # [3, img_size, img_size]

        # Resize masks to img_size
        resized_masks = []
        for m in gt_masks:
            m_resized = cv2.resize(m.astype(np.float32),
                                   (self.img_size, self.img_size),
                                   interpolation=cv2.INTER_NEAREST)
            resized_masks.append(m_resized)

        # Pad/truncate to max_instances
        num_inst = min(len(resized_masks), self.max_instances)
        padded_masks = np.zeros((self.max_instances, self.img_size, self.img_size),
                                dtype=np.float32)
        padded_classes = np.full(self.max_instances, NUM_CLASSES, dtype=np.int64)  # no-object class
        for i in range(num_inst):
            padded_masks[i] = resized_masks[i]
            padded_classes[i] = gt_classes[i]

        data_dict = {
            'roi_rgb': torch.as_tensor(roi_rgb, dtype=torch.float32),
            'gt_masks': torch.as_tensor(padded_masks, dtype=torch.float32),
            'gt_classes': torch.as_tensor(padded_classes, dtype=torch.int64),
            'num_instances': torch.as_tensor(num_inst, dtype=torch.int64),
            'image_id': torch.as_tensor(img_id, dtype=torch.int64),
        }

        # Full pipeline: also load depth and pose info
        if self.mode == 'full':
            depth_path = os.path.join(
                self.data_dir, 'depth',
                os.path.splitext(img_info['file_name'])[0] + '.exr'
            )
            if os.path.exists(depth_path):
                depth = cv2.imread(depth_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
                if depth is not None:
                    if len(depth.shape) == 3:
                        depth = depth[:, :, 0]
                    depth_resized = cv2.resize(depth, (self.img_size, self.img_size),
                                               interpolation=cv2.INTER_NEAREST)
                    data_dict['depth'] = torch.as_tensor(depth_resized, dtype=torch.float32)

            # Extract per-instance pose data if available in annotations
            for i in range(num_inst):
                extra = ann_extras[i]
                if 'rotation' in extra and 'translation' in extra:
                    # Store first instance's pose for simplicity
                    # Full pipeline would loop over instances at inference time
                    data_dict.setdefault('rotations', []).append(
                        torch.as_tensor(extra['rotation'], dtype=torch.float32))
                    data_dict.setdefault('translations', []).append(
                        torch.as_tensor(extra['translation'], dtype=torch.float32))
                if 'bbox_side_len' in extra:
                    data_dict.setdefault('bbox_side_lens', []).append(
                        torch.as_tensor(extra['bbox_side_len'], dtype=torch.float32))

            # Camera intrinsics (if provided in image info)
            if 'intrinsics' in img_info:
                data_dict['intrinsics'] = torch.as_tensor(
                    img_info['intrinsics'], dtype=torch.float32)

        return data_dict

    @staticmethod
    def _rgb_transform(rgb):
        """ImageNet normalization: [H, W, 3] uint8 -> [3, H, W] float32."""
        rgb_ = np.transpose(rgb, (2, 0, 1)).astype(np.float32) / 255.0
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        for i in range(3):
            rgb_[i] = (rgb_[i] - mean[i]) / std[i]
        return rgb_


# ---------------------------------------------------------------------------
# DataLoader helpers
# ---------------------------------------------------------------------------

def collate_nuclear(batch):
    """Custom collate that handles variable-length gt_classes/gt_masks as lists."""
    elem = batch[0]
    result = {}
    for key in elem:
        if key in ('gt_classes', 'gt_masks'):
            # These are already padded to max_instances, stack normally
            result[key] = torch.stack([d[key] for d in batch])
        elif key in ('rotations', 'translations', 'bbox_side_lens'):
            # Keep as list of lists
            result[key] = [d.get(key, []) for d in batch]
        elif isinstance(elem[key], torch.Tensor):
            result[key] = torch.stack([d[key] for d in batch])
        else:
            result[key] = [d[key] for d in batch]
    return result


def get_nuclear_data_loaders(cfg, data_type=('train', 'val')):
    """Create data loaders for nuclear workpiece dataset.

    Args:
        cfg: Config namespace with nuclear_data_path, annotation_file, etc.
        data_type: Tuple of splits to load.

    Returns:
        dict with keys like 'train_loader', 'val_loader'.
    """
    loaders = {}
    for split in data_type:
        ann_file = os.path.join(cfg.nuclear_data_path, 'annotations', f'{split}.json')
        if not os.path.exists(ann_file):
            print(f"[Warning] Annotation file not found: {ann_file}, skipping {split}")
            continue
        dataset = NuclearWorkpieceDataset(
            cfg=cfg,
            data_dir=cfg.nuclear_data_path,
            annotation_file=ann_file,
            mode='seg' if cfg.agent_type == 'segmentation' else 'full',
            img_size=cfg.img_size,
        )
        loader = data.DataLoader(
            dataset,
            batch_size=cfg.batch_size,
            shuffle=(split == 'train'),
            num_workers=cfg.num_workers,
            collate_fn=collate_nuclear,
            drop_last=(split == 'train'),
            pin_memory=True,
        )
        loaders[f'{split}_loader'] = loader
    return loaders


def process_batch_seg(batch_sample, device):
    """Move segmentation batch to device.

    Args:
        batch_sample: Dict from NuclearWorkpieceDataset collate.
        device: Target torch device.

    Returns:
        data dict with tensors on device and gt_classes/gt_masks as lists.
    """
    data = {}
    for key in batch_sample:
        if isinstance(batch_sample[key], torch.Tensor):
            data[key] = batch_sample[key].to(device)
        else:
            data[key] = batch_sample[key]

    # Convert padded gt_masks/gt_classes to per-image lists (trimmed by num_instances)
    if 'num_instances' in data:
        bs = data['num_instances'].shape[0]
        gt_masks_list = []
        gt_classes_list = []
        for b in range(bs):
            n = data['num_instances'][b].item()
            gt_masks_list.append(data['gt_masks'][b, :n])     # [n, H, W]
            gt_classes_list.append(data['gt_classes'][b, :n])  # [n]
        data['gt_masks'] = gt_masks_list
        data['gt_classes'] = gt_classes_list

    return data
