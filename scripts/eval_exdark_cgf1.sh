#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

CONFIG="${CONFIG:-configs/exdark_llmdet_swin_t_eval.py}"
CHECKPOINT="${1:-work_dirs/exdark_llmdet_swin_t_finetune/best_coco_bbox_mAP_iter_25000.pth}"
CGF1_SCORE_THRESHOLD="${2:-0.5}"
GPU_ID="${GPU_ID:-0}"
WORK_DIR="${WORK_DIR:-work_dirs/exdark_llmdet_swin_t_eval_cgf1}"

if [[ ! -f "${CONFIG}" ]]; then
    echo "Config not found: ${CONFIG}" >&2
    exit 1
fi

if [[ ! -f "${CHECKPOINT}" ]]; then
    echo "Checkpoint not found: ${CHECKPOINT}" >&2
    exit 1
fi

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"

python -u mmdet_test.py \
    "${CONFIG}" \
    "${CHECKPOINT}" \
    --work-dir "${WORK_DIR}" \
    --cfg-options \
        test_dataloader.batch_size=1 \
        test_dataloader.num_workers=0 \
        test_dataloader.persistent_workers=False \
        test_evaluator.cgf1_score_threshold="${CGF1_SCORE_THRESHOLD}"
