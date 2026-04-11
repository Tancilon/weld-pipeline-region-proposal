#!/bin/bash
# Canonical single-agent nuclear segmentation visualization entrypoint.
#
# Update the variables below for a local run, then execute:
#   bash scripts/visualize_seg_single_agent.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

: "${NUCLEAR_DATA_PATH:=./data/aiws5.2-dataset}"
: "${SEG_CKPT:=./results/ckpts/SegNet/best.pth}"
: "${SPLIT:=val}"
: "${OUTPUT_DIR:=./results/vis_seg_single_agent}"
: "${NUM_VIS:=20}"
: "${SCORE_THRESHOLD:=0.5}"
: "${IMAGE_NAMES:=}"
: "${IMG_SIZE:=224}"
: "${CUDA_VISIBLE_DEVICES:=0}"

export CUDA_VISIBLE_DEVICES

args=(
    python
    runners/visualize_seg_single_agent.py
    --nuclear_data_path "${NUCLEAR_DATA_PATH}"
    --seg_ckpt "${SEG_CKPT}"
    --split "${SPLIT}"
    --output_dir "${OUTPUT_DIR}"
    --num_vis "${NUM_VIS}"
    --score_threshold "${SCORE_THRESHOLD}"
)

if [[ -n "${IMAGE_NAMES}" ]]; then
    args+=(--image_names "${IMAGE_NAMES}")
fi

args+=(--img_size "${IMG_SIZE}")

"${args[@]}"
