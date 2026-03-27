from __future__ import annotations

import os
import sys

import torch
import torch.nn.functional as F
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from configs.config import get_config
from datasets.datasets_frontend import get_frontend_data_loaders_from_cfg
from networks.rgb_frontend import DinoV2InstanceFrontend, build_frontend_from_config
from utils.misc import exists_or_mkdir


def train_epoch(model: DinoV2InstanceFrontend, loader, optimizer, device: str):
    model.train()
    total_loss = 0.0
    for batch in tqdm(loader):
        roi_rgb = batch["roi_rgb"].to(device)
        roi_mask = batch["roi_mask"].to(device)
        class_label = batch["class_label"].to(device)
        outputs = model.forward_train_batch(roi_rgb)
        loss = torch.tensor(0.0, device=device)
        if "mask_logits" in outputs:
            loss = loss + F.binary_cross_entropy_with_logits(
                outputs["mask_logits"], roi_mask
            )
        if "score_logits" in outputs:
            score_target = torch.ones_like(outputs["score_logits"])
            loss = loss + F.binary_cross_entropy_with_logits(
                outputs["score_logits"], score_target
            )
        if "class_logits" in outputs:
            loss = loss + F.cross_entropy(outputs["class_logits"], class_label)
        if "instance_class_logits" in outputs:
            loss = loss + F.cross_entropy(outputs["instance_class_logits"], class_label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item())
    return total_loss / max(len(loader), 1)


@torch.no_grad()
def eval_epoch(model: DinoV2InstanceFrontend, loader, device: str):
    model.eval()
    total_loss = 0.0
    for batch in loader:
        roi_rgb = batch["roi_rgb"].to(device)
        roi_mask = batch["roi_mask"].to(device)
        class_label = batch["class_label"].to(device)
        outputs = model.forward_train_batch(roi_rgb)
        loss = torch.tensor(0.0, device=device)
        if "mask_logits" in outputs:
            loss = loss + F.binary_cross_entropy_with_logits(
                outputs["mask_logits"], roi_mask
            )
        if "score_logits" in outputs:
            score_target = torch.ones_like(outputs["score_logits"])
            loss = loss + F.binary_cross_entropy_with_logits(
                outputs["score_logits"], score_target
            )
        if "class_logits" in outputs:
            loss = loss + F.cross_entropy(outputs["class_logits"], class_label)
        if "instance_class_logits" in outputs:
            loss = loss + F.cross_entropy(outputs["instance_class_logits"], class_label)
        total_loss += float(loss.item())
    return total_loss / max(len(loader), 1)


def save_frontend_checkpoint(
    model: DinoV2InstanceFrontend, optimizer, epoch: int, path: str
):
    exists_or_mkdir(os.path.dirname(path))
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        path,
    )


def main():
    cfg = get_config()
    cfg.frontend_mode = "dinov2_scaffold"
    loaders = get_frontend_data_loaders_from_cfg(cfg, data_type=["train", "val"])
    model = build_frontend_from_config(cfg)
    if not isinstance(model, DinoV2InstanceFrontend):
        raise ValueError("Frontend training requires DinoV2InstanceFrontend")
    model = model.to(cfg.device)
    optimizer = torch.optim.Adam(
        [param for param in model.parameters() if param.requires_grad],
        lr=cfg.lr,
    )

    best_val = float("inf")
    for epoch in range(cfg.n_epochs):
        train_loss = train_epoch(model, loaders["train_loader"], optimizer, cfg.device)
        val_loss = eval_epoch(model, loaders["val_loader"], cfg.device)
        print(f"epoch={epoch} train_loss={train_loss:.6f} val_loss={val_loss:.6f}")
        save_frontend_checkpoint(
            model,
            optimizer,
            epoch,
            os.path.join("results", "ckpts", "Frontend", "latest.pth"),
        )
        if val_loss < best_val:
            best_val = val_loss
            save_frontend_checkpoint(
                model,
                optimizer,
                epoch,
                os.path.join("results", "ckpts", "Frontend", "best.pth"),
            )


if __name__ == "__main__":
    main()
