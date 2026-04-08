#!/bin/bash
# Canonical single-agent nuclear segmentation training entrypoint.
#
# Update the variables below for a local run, then execute:
#   bash scripts/train_seg_single_agent.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

: "${EXP_NAME:=SegNet}"
: "${NUCLEAR_DATA_PATH:=./data/aiws5.2-dataset}"
: "${POSE_INIT_CKPT:=./results/ckpts/ScoreNet/scorenet.pth}"
: "${WANDB_MODE:=online}"
: "${WANDB_PROJECT:=nuclear-seg-single-agent}"
: "${WANDB_ENTITY:=}"
: "${WANDB_RUN_NAME:=}"

: "${BATCH_SIZE:=8}"
: "${N_EPOCHS:=100}"
: "${NUM_WORKERS:=4}"
: "${IMG_SIZE:=224}"
: "${CUDA_VISIBLE_DEVICES:=0}"

export CUDA_VISIBLE_DEVICES

python scripts/check_nuclear_seg_dataset.py --nuclear_data_path "${NUCLEAR_DATA_PATH}"

args=(
    python
    runners/trainer.py
    --dataset_type nuclear
    --nuclear_data_path "${NUCLEAR_DATA_PATH}"
    --agent_type segmentation
    --enable_segmentation
    --num_queries 50
    --query_inject_layer -4
    --num_object_classes 6
    --unfreeze_dino_last_n 4
    --dino pointwise
    --batch_size "${BATCH_SIZE}"
    --n_epochs "${N_EPOCHS}"
    --lr 1e-4
    --lr_decay 0.99
    --warmup 50
    --eval_freq 10
    --log_dir "${EXP_NAME}"
    --is_train
    --img_size "${IMG_SIZE}"
    --num_workers "${NUM_WORKERS}"
    --wandb_mode "${WANDB_MODE}"
    --wandb_project "${WANDB_PROJECT}"
)

if [[ -n "${POSE_INIT_CKPT}" ]]; then
    args+=(--pretrained_score_model_path "${POSE_INIT_CKPT}")
fi

if [[ -n "${WANDB_ENTITY}" ]]; then
    args+=(--wandb_entity "${WANDB_ENTITY}")
fi

if [[ -n "${WANDB_RUN_NAME}" ]]; then
    args+=(--wandb_run_name "${WANDB_RUN_NAME}")
fi

"${args[@]}"
