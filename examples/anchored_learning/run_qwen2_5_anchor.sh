#!/usr/bin/env bash
set -euo pipefail

# Example: paper-style two-stage Anchored Learning on KDFlow.
#
# Environment variables you will usually override:
#   BASE_MODEL=Qwen/Qwen2.5-3B-Instruct
#   RAW_JSON=data/train.json
#   WORK_DIR=outputs/anchored_qwen
#   NUM_GPUS=1
#   GLOBAL_BSZ=16
#   MICRO_BSZ=1

BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-3B-Instruct}"
RAW_JSON="${RAW_JSON:-data/train.json}"
WORK_DIR="${WORK_DIR:-outputs/anchored_qwen}"
TRAIN_JSON="${WORK_DIR}/train.kdflow.json"
SFT_DIR="${WORK_DIR}/sft_ref"
ANCHOR_DIR="${WORK_DIR}/anchored"

NUM_NODES="${NUM_NODES:-1}"
NUM_GPUS="${NUM_GPUS:-1}"
TEACHER_TP_SIZE="${TEACHER_TP_SIZE:-1}"
GLOBAL_BSZ="${GLOBAL_BSZ:-16}"
MICRO_BSZ="${MICRO_BSZ:-1}"
MAX_LEN="${MAX_LEN:-4096}"
LR="${LR:-1e-5}"
SFT_EPOCHS="${SFT_EPOCHS:-3}"

ANCHOR_ALPHA="${ANCHOR_ALPHA:-0.5}"
ANCHOR_OUTER_ITERS="${ANCHOR_OUTER_ITERS:-5}"
ANCHOR_INNER_EPOCHS="${ANCHOR_INNER_EPOCHS:-5}"
ANCHOR_TOTAL_EPOCHS=$((ANCHOR_OUTER_ITERS * ANCHOR_INNER_EPOCHS))

mkdir -p "${WORK_DIR}"

python scripts/prepare_alpaca_json.py \
  --input_file "${RAW_JSON}" \
  --output_file "${TRAIN_JSON}" \
  --template simple

# Stage 1: train fixed p_sft from p_base.
python -m kdflow.cli.train_sft \
  --student_name_or_path "${BASE_MODEL}" \
  --train_dataset_path "${TRAIN_JSON}" \
  --input_key input \
  --output_key output \
  --max_len "${MAX_LEN}" \
  --num_nodes "${NUM_NODES}" \
  --num_gpus_per_node "${NUM_GPUS}" \
  --num_epochs "${SFT_EPOCHS}" \
  --train_batch_size "${GLOBAL_BSZ}" \
  --micro_train_batch_size "${MICRO_BSZ}" \
  --learning_rate "${LR}" \
  --bf16 \
  --gradient_checkpointing \
  --chunked_loss_size 2048 \
  --save_path "${SFT_DIR}"

# Stage 2: initialize p_theta^(0) from p_base; use p_sft as fixed teacher.
# The patch refreshes the outer snapshot every ANCHOR_INNER_EPOCHS epochs.
python -m kdflow.cli.train_kd_off_policy \
  --student_name_or_path "${BASE_MODEL}" \
  --teacher_name_or_path "${SFT_DIR}" \
  --train_dataset_path "${TRAIN_JSON}" \
  --input_key input \
  --output_key output \
  --max_len "${MAX_LEN}" \
  --num_nodes "${NUM_NODES}" \
  --num_gpus_per_node "${NUM_GPUS}" \
  --teacher_tp_size "${TEACHER_TP_SIZE}" \
  --teacher_pp_size 1 \
  --teacher_dp_size "${NUM_GPUS}" \
  --kd_algorithm anchored_kd \
  --kd_loss_fn kl \
  --kd_ratio 1.0 \
  --anchor_alpha "${ANCHOR_ALPHA}" \
  --anchor_interpolation logit \
  --anchor_inner_epochs "${ANCHOR_INNER_EPOCHS}" \
  --anchor_snapshot_mode model \
  --anchor_temperature 1.0 \
  --num_epochs "${ANCHOR_TOTAL_EPOCHS}" \
  --train_batch_size "${GLOBAL_BSZ}" \
  --micro_train_batch_size "${MICRO_BSZ}" \
  --learning_rate "${LR}" \
  --bf16 \
  --gradient_checkpointing \
  --chunked_loss_size 2048 \
  --save_path "${ANCHOR_DIR}"
