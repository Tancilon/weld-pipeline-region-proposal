#!/bin/bash
# Train EoMT segmentation/classification head on nuclear workpiece dataset.
#
# Prerequisites:
#   1. Prepare COCO-format annotations under NUCLEAR_DATA_PATH/annotations/{train,val}.json
#   2. Place images under NUCLEAR_DATA_PATH/images/
#   3. (Optional) Pre-trained score model weights for pose backbone initialization
#
# Usage:
#   bash scripts/train_seg.sh

NUCLEAR_DATA_PATH="./data/aiws5.2_nuclear_workpieces"
PRETRAINED_SCORE_MODEL=""  # e.g. "./results/ckpts/ScoreNet/ckpt_epoch50.pth"

CUDA_VISIBLE_DEVICES=0 python runners/trainer.py \
    --dataset_type nuclear \
    --nuclear_data_path ${NUCLEAR_DATA_PATH} \
    --agent_type segmentation \
    --enable_segmentation \
    --num_queries 50 \
    --query_inject_layer -4 \
    --num_object_classes 6 \
    --unfreeze_dino_last_n 4 \
    --dino pointwise \
    --batch_size 8 \
    --n_epochs 100 \
    --lr 1e-4 \
    --lr_decay 0.99 \
    --warmup 50 \
    --eval_freq 10 \
    --log_dir SegNet \
    --is_train \
    --img_size 224 \
    --num_workers 4 \
    ${PRETRAINED_SCORE_MODEL:+--pretrained_score_model_path ${PRETRAINED_SCORE_MODEL}}
