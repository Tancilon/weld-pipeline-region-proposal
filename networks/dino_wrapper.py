"""
DINOv2 wrapper with learnable query token injection for EoMT-style segmentation.

Based on "Your ViT is Secretly an Image Segmentation Model" (CVPR 2025).
The backbone is split at `inject_layer` into a frozen pose tail from the
original DINO model and a deep-copied segmentation tail that receives query
tokens. Pose features are taken from the original frozen tail, while the
segmentation path uses only the cloned tail. The wrapper owns the trainability
contract: the original DINO path stays frozen, and query/segmentation modules
remain trainable regardless of the caller's input model state.
"""

import copy
import torch
import torch.nn as nn


class DINOv2WithQueries(nn.Module):
    """Wraps a DINOv2 ViT model to support learnable query token injection.

    The forward pass splits the transformer blocks into two stages:
      - Stage 1 (layers 0..inject_layer-1): shared frozen stem.
      - Stage 2 (layers inject_layer..end): original frozen pose tail for pose
        features, cloned trainable segmentation tail for query injection.

    Args:
        dino_model: A DINOv2 model loaded via torch.hub (e.g. dinov2_vits14).
        num_query_tokens: Number of learnable query tokens to inject.
        query_inject_layer: Layer index (negative = from end) at which to
            inject queries. Default -4 means the last 4 layers.
    """

    def __init__(self, dino_model, num_query_tokens=50, query_inject_layer=-4):
        super().__init__()
        self.dino = dino_model
        self.num_query_tokens = num_query_tokens
        self.embed_dim = dino_model.embed_dim  # 384 for ViT-S

        # Resolve negative layer index
        num_blocks = len(dino_model.blocks)
        if query_inject_layer < 0:
            query_inject_layer = num_blocks + query_inject_layer
        assert 0 < query_inject_layer < num_blocks, \
            f"query_inject_layer={query_inject_layer} out of range [1, {num_blocks-1}]"
        self.inject_layer = query_inject_layer

        # Learnable query embeddings
        # shape: [num_query_tokens, embed_dim]
        self.query_embed = nn.Embedding(num_query_tokens, self.embed_dim)
        nn.init.trunc_normal_(self.query_embed.weight, std=0.02)

        self.seg_blocks = copy.deepcopy(self.dino.blocks[self.inject_layer:])
        if hasattr(self.dino, 'norm') and self.dino.norm is not None:
            self.seg_norm = copy.deepcopy(self.dino.norm)
        else:
            self.seg_norm = None

        self._freeze_pose_path()
        self._mark_segmentation_path_trainable()

    def _freeze_pose_path(self):
        self.dino.requires_grad_(False)

    def _mark_segmentation_path_trainable(self):
        self.query_embed.requires_grad_(True)
        self.seg_blocks.requires_grad_(True)
        if self.seg_norm is not None:
            self.seg_norm.requires_grad_(True)

    def _run_stem(self, x):
        tokens = self._prepare_tokens(x)
        for blk in self.dino.blocks[:self.inject_layer]:
            tokens = blk(tokens)
        return tokens

    def _run_pose_tail(self, tokens):
        pose_tokens = tokens
        for blk in self.dino.blocks[self.inject_layer:]:
            pose_tokens = blk(pose_tokens)
        if hasattr(self.dino, 'norm') and self.dino.norm is not None:
            pose_tokens = self.dino.norm(pose_tokens)
        return pose_tokens

    def _run_seg_tail(self, tokens):
        seg_tokens = tokens
        for blk in self.seg_blocks:
            seg_tokens = blk(seg_tokens)
        if self.seg_norm is not None:
            seg_tokens = self.seg_norm(seg_tokens)
        return seg_tokens

    def _prepare_tokens(self, x):
        """Run DINOv2 patch embedding + CLS token + positional embedding.

        Returns the full token sequence [CLS, (registers), patches] ready
        for the transformer blocks.
        """
        # DINOv2 prepare_tokens_with_masks handles:
        # 1. patch_embed(x) -> [bs, num_patches, embed_dim]
        # 2. prepend cls_token
        # 3. add pos_embed (interpolated if needed)
        # 4. prepend register_tokens (if model has them)
        tokens = self.dino.prepare_tokens_with_masks(x)
        return tokens

    def _count_prefix_tokens(self):
        """Count number of non-patch prefix tokens (CLS + registers)."""
        n = 1  # CLS token is always present
        if hasattr(self.dino, 'register_tokens') and self.dino.register_tokens is not None:
            n += self.dino.register_tokens.shape[1] # register token shape: [1, num_registers, embed_dim]
        return n

    def forward_with_queries(self, x):
        """Forward pass with query injection.

        Args:
            x: Input images [bs, 3, H, W] (H, W divisible by 14).

        Returns:
            patch_tokens_for_pose: [bs, num_patches, embed_dim]
                Patch tokens after the full frozen pose tail.
            query_tokens: [bs, num_query_tokens, embed_dim]
                Query token outputs after joint attention with patches.
            patch_tokens_seg: [bs, num_patches, embed_dim]
                Patch tokens after joint attention (for mask dot-product).
        """
        # Prepare token sequence: [CLS, (registers), patches]
        tokens = self._run_stem(x)  # [B, N_total, D]
        num_prefix = self._count_prefix_tokens()
        bs = x.shape[0]

        # Pose branch: run the original frozen DINO tail without query tokens.
        pose_tokens = self._run_pose_tail(tokens)
        patch_tokens_for_pose = pose_tokens[:, num_prefix:, :].detach().clone()

        # Segmentation branch: inject queries into the cloned tail only.
        query_out, patch_out = self._forward_segmentation_tokens(tokens, num_prefix)

        return patch_tokens_for_pose, query_out, patch_out

    def _forward_segmentation_tokens(self, tokens, num_prefix):
        bs = tokens.shape[0]
        query_tokens = self.query_embed.weight.unsqueeze(0).expand(bs, -1, -1)
        tokens = torch.cat([tokens, query_tokens], dim=1)
        tokens = self._run_seg_tail(tokens)

        # Separate outputs
        num_queries = self.num_query_tokens
        query_out = tokens[:, -num_queries:, :]           # [bs, N, embed_dim]
        patch_out = tokens[:, num_prefix:-num_queries, :]  # [bs, num_patches, embed_dim]

        return query_out, patch_out

    def forward_segmentation(self, x):
        """Forward pass for segmentation only.

        Runs the shared stem and the cloned segmentation tail. The original
        frozen pose tail is skipped.
        """
        tokens = self._run_stem(x)
        num_prefix = self._count_prefix_tokens()
        return self._forward_segmentation_tokens(tokens, num_prefix)

    def forward_patch_only(self, x): # [B, 3, H, W]
        """Standard DINOv2 forward without query injection.

        Used when segmentation is disabled or for compatibility.
        Returns patch tokens from the last intermediate layer (same as
        dino.get_intermediate_layers(x)[0]).
        """
        return self.dino.get_intermediate_layers(x)[0] # [B, N_patch, D]， D = 384 for ViT-S
