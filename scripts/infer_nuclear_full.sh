#!/bin/bash
# Canonical single-agent nuclear full-pipeline inference entrypoint.
#
# Update the variables below for a local run, then execute:
#   bash scripts/infer_nuclear_full.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

: "${NUCLEAR_DATA_PATH:=./data/aiws5.2_nuclear_workpieces}"
: "${SEG_CKPT:=./results/ckpts/SegNet/best.pth}"
: "${ENERGY_CKPT:=./results/ckpts/EnergyNet/energynet.pth}"
: "${SCALE_CKPT:=./results/ckpts/ScaleNet/scalenet.pth}"
: "${SPLIT:=val}"
: "${OUTPUT_DIR:=./results/full_pipeline_segnet_best}"
: "${NUM_VIS:=20}"
: "${SCORE_THRESHOLD:=0.5}"
: "${REPEAT_NUM:=10}"
: "${NUM_POINTS:=1024}"
: "${IMG_SIZE:=224}"
: "${CUDA_VISIBLE_DEVICES:=0}"

export CUDA_VISIBLE_DEVICES

args=(
    python
    runners/infer_nuclear_full.py
    --nuclear_data_path "${NUCLEAR_DATA_PATH}"
    --seg_ckpt "${SEG_CKPT}"
    --energy_ckpt "${ENERGY_CKPT}"
    --scale_ckpt "${SCALE_CKPT}"
    --split "${SPLIT}"
    --output_dir "${OUTPUT_DIR}"
    --num_vis "${NUM_VIS}"
    --score_threshold "${SCORE_THRESHOLD}"
    --repeat_num "${REPEAT_NUM}"
    --num_points "${NUM_POINTS}"
    --img_size "${IMG_SIZE}"
)

"${args[@]}"
