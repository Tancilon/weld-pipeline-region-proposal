from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class DinoFeatureOutput:
    patch_tokens: torch.Tensor | None
    global_feat: torch.Tensor | None
    feature_map: torch.Tensor | None


class DinoV2Backbone(nn.Module):
    def __init__(
        self,
        model_name: str = "dinov2_vits14",
        freeze: bool = True,
        device: str = "cuda",
    ) -> None:
        super().__init__()
        self.model_name = model_name
        self.freeze = freeze
        self.device_name = device
        self.model: nn.Module = torch.hub.load("facebookresearch/dinov2", model_name)
        self.model = self.model.to(device)
        if freeze:
            self.model.requires_grad_(False)
        self.embed_dim = self._infer_embed_dim()
        self.patch_size = 14

    def forward(
        self,
        rgb: torch.Tensor,
        return_global: bool = True,
        return_dense: bool = True,
    ) -> DinoFeatureOutput:
        rgb = rgb.to(self.device_name)
        patch_tokens = None
        feature_map = None
        if return_dense:
            patch_tokens = self.model.get_intermediate_layers(rgb)[0]
            feature_map = self._tokens_to_feature_map(
                patch_tokens, rgb.shape[-2], rgb.shape[-1]
            )

        global_feat = None
        if return_global:
            global_feat = self.model(rgb)

        return DinoFeatureOutput(
            patch_tokens=patch_tokens,
            global_feat=global_feat,
            feature_map=feature_map,
        )

    def _infer_embed_dim(self) -> int:
        embed_dim = getattr(self.model, "embed_dim", None)
        if isinstance(embed_dim, int):
            return embed_dim

        dummy = torch.zeros(1, 3, 224, 224, device=self.device_name)
        with torch.no_grad():
            tokens = self.model.get_intermediate_layers(dummy)[0]
        return int(tokens.shape[-1])

    def _tokens_to_feature_map(
        self, patch_tokens: torch.Tensor, height: int, width: int
    ) -> torch.Tensor:
        h_tokens = height // self.patch_size
        w_tokens = width // self.patch_size
        return (
            patch_tokens.view(
                patch_tokens.shape[0], h_tokens, w_tokens, patch_tokens.shape[-1]
            )
            .permute(0, 3, 1, 2)
            .contiguous()
        )
