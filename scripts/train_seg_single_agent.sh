#!/bin/bash
# Canonical single-agent nuclear segmentation training entrypoint.
#
# Update the variables below for a local run, then execute:
#   bash scripts/train_seg_single_agent.sh

set -euo pipefail

EXP_NAME="SegNet"
NUCLEAR_DATA_PATH="./data/aiws5.2_nuclear_workpieces"
POSE_INIT_CKPT=""
WANDB_MODE="online"
WANDB_PROJECT="nuclear-seg-single-agent"
WANDB_ENTITY=""

BATCH_SIZE=8
N_EPOCHS=100
NUM_WORKERS=4
IMG_SIZE=224

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
WANDB_MODE="${WANDB_MODE}" \
WANDB_PROJECT="${WANDB_PROJECT}" \
WANDB_ENTITY="${WANDB_ENTITY}" \
args=(
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
    --wandb_entity "${WANDB_ENTITY}"
)

if [[ -n "${POSE_INIT_CKPT}" ]]; then
    args+=(--pretrained_score_model_path "${POSE_INIT_CKPT}")
fi

python "${args[@]}"
