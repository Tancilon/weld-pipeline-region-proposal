"""
EoMT segmentation and classification head.

Implements the lightweight head from "Your ViT is Secretly an Image
Segmentation Model" (CVPR 2025): a linear class head, a 3-layer MLP mask
head with dot-product mask prediction, upscaling blocks, and the training
criterion with Hungarian matching.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.optimize import linear_sum_assignment


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class ScaleBlock(nn.Module):
    """2x bilinear upsample followed by Conv + GroupNorm + GELU."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm = nn.GroupNorm(min(8, out_channels), out_channels)
        self.act = nn.GELU()

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        return self.act(self.norm(self.conv(x)))


# ---------------------------------------------------------------------------
# EoMT Head
# ---------------------------------------------------------------------------

class EoMTHead(nn.Module):
    """Segmentation + classification head operating on query/patch tokens.

    Args:
        embed_dim: Token embedding dimension (384 for ViT-S).
        num_classes: Number of object categories (excluding *no-object*).
        num_queries: Number of query tokens (must match DINOv2WithQueries).
        mask_head_hidden: Hidden dimension of the mask MLP.
        num_upscale_blocks: Number of 2x upscaling blocks applied to the
            low-resolution mask logits.
    """

    def __init__(self, embed_dim=384, num_classes=6, num_queries=50,
                 mask_head_hidden=384, num_upscale_blocks=2):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.num_queries = num_queries

        # Classification head: query -> (num_classes + 1) logits
        self.class_head = nn.Linear(embed_dim, num_classes + 1)

        # Mask head: 3-layer MLP projecting query tokens for dot-product
        self.mask_head = nn.Sequential(
            nn.Linear(embed_dim, mask_head_hidden),
            nn.GELU(),
            nn.Linear(mask_head_hidden, mask_head_hidden),
            nn.GELU(),
            nn.Linear(mask_head_hidden, embed_dim),
        )

        # Upscaling pathway: progressively upsample patch spatial features,
        # then dot-product with projected query embeddings at higher resolution.
        channels = [embed_dim]
        for i in range(num_upscale_blocks):
            channels.append(max(embed_dim // (2 ** (i + 1)), 32))
        self.upscale_blocks = nn.ModuleList([
            ScaleBlock(channels[i], channels[i + 1])
            for i in range(num_upscale_blocks)
        ])
        self.upscaled_dim = channels[-1]

        # Project query mask embeddings to match upscaled feature dim
        self.mask_up_proj = nn.Linear(embed_dim, self.upscaled_dim)

    def forward(self, query_tokens, patch_tokens, patch_hw=(16, 16),
                img_size=(224, 224)):
        """Predict class logits and mask logits.

        Args:
            query_tokens: [bs, N, embed_dim] from DINOv2WithQueries.
            patch_tokens: [bs, num_patches, embed_dim] (after query injection).
            patch_hw: Spatial layout of patch tokens (H_p, W_p).
            img_size: Target output mask resolution (H, W).

        Returns:
            class_logits: [bs, N, num_classes+1]
            mask_logits:  [bs, N, H, W]
        """
        bs, nq, _ = query_tokens.shape
        ph, pw = patch_hw

        # --- Classification ---
        class_logits = self.class_head(query_tokens)  # [bs, N, C+1]

        # --- Mask prediction ---
        mask_embed = self.mask_head(query_tokens)  # [bs, N, embed_dim]

        # Reshape patch tokens to spatial grid and upsample
        patch_spatial = patch_tokens.reshape(bs, ph, pw, self.embed_dim)
        patch_spatial = patch_spatial.permute(0, 3, 1, 2)  # [bs, D, ph, pw]
        for block in self.upscale_blocks:
            patch_spatial = block(patch_spatial)  # [bs, D', H', W']

        # Project query embeddings to match upscaled dim, then dot-product
        mask_embed_up = self.mask_up_proj(mask_embed)  # [bs, N, D']
        mask_logits = torch.einsum('bqd,bdhw->bqhw', mask_embed_up, patch_spatial)

        # Interpolate to target image size if needed
        if mask_logits.shape[-2:] != tuple(img_size):
            mask_logits = F.interpolate(
                mask_logits, size=img_size, mode='bilinear', align_corners=False
            )

        return class_logits, mask_logits


# ---------------------------------------------------------------------------
# Loss / Criterion
# ---------------------------------------------------------------------------

def dice_loss(pred_masks, gt_masks):
    """Compute per-pair dice loss.

    Args:
        pred_masks: [N, H*W] sigmoid probabilities.
        gt_masks:   [M, H*W] binary ground truth.

    Returns:
        loss_matrix: [N, M] dice loss for each pred-gt pair.
    """
    # Numerator: 2 * |pred ∩ gt|
    numerator = 2.0 * torch.einsum('nh,mh->nm', pred_masks, gt_masks)
    # Denominator: |pred| + |gt|
    denom = pred_masks.sum(dim=-1).unsqueeze(1) + gt_masks.sum(dim=-1).unsqueeze(0)
    return 1.0 - (numerator + 1.0) / (denom + 1.0)


def sigmoid_bce_loss(pred_masks, gt_masks):
    """Compute per-pair binary cross-entropy (with logits already sigmoided).

    Args:
        pred_masks: [N, H*W] probabilities (already sigmoided).
        gt_masks:   [M, H*W] binary ground truth.

    Returns:
        loss_matrix: [N, M] mean BCE for each pred-gt pair.
    """
    N, HW = pred_masks.shape
    M = gt_masks.shape[0]
    # Expand for pairwise computation
    pred_exp = pred_masks.unsqueeze(1).expand(N, M, HW)  # [N, M, HW]
    gt_exp = gt_masks.unsqueeze(0).expand(N, M, HW)      # [N, M, HW]
    loss = F.binary_cross_entropy(pred_exp, gt_exp, reduction='none')
    return loss.mean(dim=-1)  # [N, M]


class EoMTCriterion(nn.Module):
    """Hungarian-matching loss for EoMT predictions.

    Combines classification cross-entropy, mask BCE, and mask Dice losses.

    Args:
        num_classes: Number of object classes (excluding no-object).
        cls_weight: Weight for classification CE loss.
        mask_weight: Weight for mask BCE loss.
        dice_weight: Weight for mask Dice loss.
        no_object_weight: CE weight for the no-object class (typically < 1
            to down-weight the abundant no-object predictions).
    """

    def __init__(self, num_classes=6, cls_weight=2.0, mask_weight=5.0,
                 dice_weight=5.0, no_object_weight=0.1):
        super().__init__()
        self.num_classes = num_classes
        self.cls_weight = cls_weight
        self.mask_weight = mask_weight
        self.dice_weight = dice_weight
        self.no_object_weight = no_object_weight

        # no-object is the last class index
        self.no_object_class = num_classes

    @torch.no_grad()
    def _hungarian_match(self, class_logits, mask_logits, gt_classes, gt_masks):
        """Compute optimal assignment between predictions and ground truth.

        Args:
            class_logits: [N, C+1] predicted class logits for one image.
            mask_logits:  [N, H, W] predicted mask logits for one image.
            gt_classes:   [M] ground-truth class indices.
            gt_masks:     [M, H, W] ground-truth binary masks.

        Returns:
            (pred_indices, gt_indices): matched index arrays.
        """
        N = class_logits.shape[0]
        M = gt_classes.shape[0]
        if M == 0:
            return (torch.tensor([], dtype=torch.long),
                    torch.tensor([], dtype=torch.long))

        # Class cost: -log P(correct class)
        class_prob = class_logits.softmax(dim=-1)  # [N, C+1]
        class_cost = -class_prob[:, gt_classes]     # [N, M]

        # Mask costs
        pred_masks_flat = mask_logits.flatten(1).sigmoid()  # [N, H*W]
        gt_masks_flat = gt_masks.flatten(1).float()         # [M, H*W]

        bce_cost = sigmoid_bce_loss(pred_masks_flat, gt_masks_flat)  # [N, M]
        dice_cost = dice_loss(pred_masks_flat, gt_masks_flat)        # [N, M]

        # Combined cost matrix
        cost = (self.cls_weight * class_cost +
                self.mask_weight * bce_cost +
                self.dice_weight * dice_cost)

        cost_np = cost.detach().cpu().numpy()
        pred_idx, gt_idx = linear_sum_assignment(cost_np)
        return (torch.as_tensor(pred_idx, dtype=torch.long),
                torch.as_tensor(gt_idx, dtype=torch.long))

    def forward(self, class_logits, mask_logits, gt_classes_list, gt_masks_list):
        """Compute losses over a batch.

        Args:
            class_logits: [bs, N, C+1] predicted class logits.
            mask_logits:  [bs, N, H, W] predicted mask logits.
            gt_classes_list: List[Tensor] of length bs, each [M_i] with class indices.
            gt_masks_list:   List[Tensor] of length bs, each [M_i, H, W] binary masks.

        Returns:
            dict with keys 'cls_loss', 'mask_loss', 'dice_loss', 'total_loss'.
        """
        bs = class_logits.shape[0]
        device = class_logits.device

        total_cls = torch.tensor(0.0, device=device)
        total_mask = torch.tensor(0.0, device=device)
        total_dice = torch.tensor(0.0, device=device)

        # CE weight vector: normal for object classes, reduced for no-object
        ce_weight = torch.ones(self.num_classes + 1, device=device)
        ce_weight[self.no_object_class] = self.no_object_weight

        for b in range(bs):
            cls_pred = class_logits[b]   # [N, C+1]
            msk_pred = mask_logits[b]    # [N, H, W]
            gt_cls = gt_classes_list[b].to(device)  # [M]
            gt_msk = gt_masks_list[b].to(device)    # [M, H, W]

            pred_idx, gt_idx = self._hungarian_match(cls_pred, msk_pred, gt_cls, gt_msk)

            # --- Classification loss ---
            # Build target: no-object for all, then assign matched
            target_classes = torch.full(
                (cls_pred.shape[0],), self.no_object_class,
                dtype=torch.long, device=device
            )
            if len(pred_idx) > 0:
                target_classes[pred_idx] = gt_cls[gt_idx]
            total_cls += F.cross_entropy(cls_pred, target_classes, weight=ce_weight)

            # --- Mask losses (only for matched pairs) ---
            if len(pred_idx) > 0:
                matched_pred = msk_pred[pred_idx].flatten(1).sigmoid()  # [K, H*W]
                matched_gt = gt_msk[gt_idx].flatten(1).float()          # [K, H*W]

                # BCE
                total_mask += F.binary_cross_entropy(
                    matched_pred, matched_gt, reduction='mean'
                )
                # Dice
                numerator = 2.0 * (matched_pred * matched_gt).sum(dim=-1)
                denom = matched_pred.sum(dim=-1) + matched_gt.sum(dim=-1)
                total_dice += (1.0 - (numerator + 1.0) / (denom + 1.0)).mean()

        # Average over batch
        total_cls /= bs
        total_mask /= bs
        total_dice /= bs

        total_loss = (self.cls_weight * total_cls +
                      self.mask_weight * total_mask +
                      self.dice_weight * total_dice)

        return {
            'cls_loss': total_cls,
            'mask_loss': total_mask,
            'dice_loss': total_dice,
            'total_loss': total_loss,
        }
