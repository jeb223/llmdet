#!/usr/bin/env bash
set -euo pipefail

# AutoDL-friendly training script for the mine15 LLMDet + SAM baseline.
# Run from anywhere:
#   bash scripts/train_mine15_llmdet_sam.sh
#
# Common overrides:
#   BATCH_SIZE=4 MAX_ITERS=100000 bash scripts/train_mine15_llmdet_sam.sh
#   RUN_SAM_EVAL=0 bash scripts/train_mine15_llmdet_sam.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${REPO_DIR:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
cd "${REPO_DIR}"

CONFIG="${CONFIG:-configs/mine15_llmdet_swin_t_finetune.py}"
DATA_ROOT="${DATA_ROOT:-data}"
SOURCE_DATA_ROOT="${SOURCE_DATA_ROOT:-${HOME}/autodl-tmp/data/data}"

LLMDET_INIT_CKPT="${LLMDET_INIT_CKPT:-weights/llmdet/tiny.pth}"
SAM_CKPT="${SAM_CKPT:-${HOME}/autodl-tmp/weights/sam/sam_vit_b_01ec64.pth}"
SAM_MODEL_TYPE="${SAM_MODEL_TYPE:-vit_b}"

WORK_DIR="${WORK_DIR:-work_dirs/mine15_llmdet_swin_t_finetune}"
BATCH_SIZE="${BATCH_SIZE:-2}"
NUM_WORKERS="${NUM_WORKERS:-4}"
MAX_ITERS="${MAX_ITERS:-100000}"
EVAL_INTERVAL="${EVAL_INTERVAL:-10000}"
BASE_BATCH_SIZE="${BASE_BATCH_SIZE:-16}"
AMP="${AMP:-1}"

RUN_SAM_EVAL="${RUN_SAM_EVAL:-1}"
QUERY_MODE="${QUERY_MODE:-per-category}"
BOX_THRESHOLD="${BOX_THRESHOLD:-0.3}"
SAM_EVAL_ROOT="${SAM_EVAL_ROOT:-${HOME}/autodl-tmp/outputs/mine15_llmdet_sam}"
EVAL_SCRIPT="${EVAL_SCRIPT:-${REPO_DIR}/evaluate_llmdet_sam_map_cgf1.py}"

echo "[1/5] Checking paths"
if [[ ! -f "${CONFIG}" ]]; then
  echo "Config not found: ${CONFIG}" >&2
  exit 1
fi

if [[ ! -e "${DATA_ROOT}" ]]; then
  if [[ -d "${SOURCE_DATA_ROOT}" ]]; then
    ln -s "${SOURCE_DATA_ROOT}" "${DATA_ROOT}"
  else
    echo "Data root not found: ${DATA_ROOT}" >&2
    echo "Also tried SOURCE_DATA_ROOT=${SOURCE_DATA_ROOT}" >&2
    exit 1
  fi
fi

if [[ ! -f "${DATA_ROOT}/train/_annotations.coco.json" ]]; then
  echo "Missing ${DATA_ROOT}/train/_annotations.coco.json" >&2
  exit 1
fi

if [[ ! -f "${DATA_ROOT}/valid/_annotations.coco.json" ]]; then
  echo "Missing ${DATA_ROOT}/valid/_annotations.coco.json" >&2
  exit 1
fi

if [[ ! -f "${LLMDET_INIT_CKPT}" ]]; then
  echo "LLMDet init checkpoint not found: ${LLMDET_INIT_CKPT}" >&2
  exit 1
fi

echo "[2/5] Preparing ODVG annotations"
if [[ ! -f "${DATA_ROOT}/odvg_label_map.json" || ! -f "${DATA_ROOT}/train/odvg_train.jsonl" ]]; then
  python scripts/convert_coco_to_odvg_15cat.py --data-root "${DATA_ROOT}"
else
  echo "ODVG annotations already exist"
fi

echo "[3/5] Training LLMDet detection branch"
TRAIN_CMD=(
  python mmdet_train.py "${CONFIG}"
  --work-dir "${WORK_DIR}"
  --cfg-options
  "load_from=${LLMDET_INIT_CKPT}"
  "train_dataloader.batch_size=${BATCH_SIZE}"
  "train_dataloader.num_workers=${NUM_WORKERS}"
  "custom_hooks.0.batch_size=${BATCH_SIZE}"
  "custom_hooks.0.num_workers=2"
  "train_cfg.max_iters=${MAX_ITERS}"
  "train_cfg.val_interval=${EVAL_INTERVAL}"
  "default_hooks.checkpoint.interval=${EVAL_INTERVAL}"
  "custom_hooks.0.interval=${EVAL_INTERVAL}"
  "auto_scale_lr.base_batch_size=${BASE_BATCH_SIZE}"
)

if [[ "${AMP}" == "1" ]]; then
  TRAIN_CMD+=(--amp)
fi

"${TRAIN_CMD[@]}"

echo "[4/5] Training records"
echo "Work dir: ${WORK_DIR}"
echo "MMDetection log: ${WORK_DIR}"
echo "Validation loss/metric JSONL: ${WORK_DIR}/train_eval_records.jsonl"

if [[ "${RUN_SAM_EVAL}" != "1" ]]; then
  echo "[5/5] Skipping SAM evaluation because RUN_SAM_EVAL=${RUN_SAM_EVAL}"
  exit 0
fi

if [[ ! -f "${SAM_CKPT}" ]]; then
  echo "SAM checkpoint not found: ${SAM_CKPT}" >&2
  exit 1
fi

if [[ ! -f "${EVAL_SCRIPT}" ]]; then
  if [[ -f "${REPO_DIR}/../evaluate_llmdet_sam_map_cgf1.py" ]]; then
    EVAL_SCRIPT="${REPO_DIR}/../evaluate_llmdet_sam_map_cgf1.py"
  else
    echo "Evaluation script not found: ${EVAL_SCRIPT}" >&2
    echo "You can override it with EVAL_SCRIPT=/path/to/evaluate_llmdet_sam_map_cgf1.py" >&2
    exit 1
  fi
fi

echo "[5/5] Evaluating saved checkpoints with SAM"
mkdir -p "${SAM_EVAL_ROOT}"

iter="${EVAL_INTERVAL}"
while [[ "${iter}" -le "${MAX_ITERS}" ]]; do
  ckpt="${WORK_DIR}/iter_${iter}.pth"
  if [[ -f "${ckpt}" ]]; then
    out_dir="${SAM_EVAL_ROOT}/iter_${iter}"
    echo "Evaluating ${ckpt} -> ${out_dir}"
    python "${EVAL_SCRIPT}" \
      --val-data-dir "${DATA_ROOT}/valid" \
      --llmdet-repo "${REPO_DIR}" \
      --llmdet-config "${CONFIG}" \
      --llmdet-checkpoint "${ckpt}" \
      --sam-checkpoint "${SAM_CKPT}" \
      --sam-model-type "${SAM_MODEL_TYPE}" \
      --query-mode "${QUERY_MODE}" \
      --box-threshold "${BOX_THRESHOLD}" \
      --output-dir "${out_dir}"
  else
    echo "Skip missing checkpoint: ${ckpt}"
  fi
  iter=$((iter + EVAL_INTERVAL))
done

echo "Done."
