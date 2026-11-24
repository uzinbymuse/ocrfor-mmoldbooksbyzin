#!/usr/bin/env bash
# =============================================================================
# Train mya_dhamma_v12 (Balanced Fine-Tuning with Golden-Set Regression Gate)
# Base Model: tessdata/mya_dhamma_v10.traineddata
# Training Data: 6,800 lines (100% UTN #11 Canonical Normalization)
# =============================================================================

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

MODEL_NAME="${MODEL_NAME:-mya_dhamma_v12}"
BASE_TRAINEDDATA="${BASE_TRAINEDDATA:-tessdata/mya_dhamma_v10.traineddata}"
BASE_CKPT="${BASE_CKPT:-checkpoints/mya_dhamma_v10_checkpoint}"
STARTER_DATA="${STARTER_DATA:-$BASE_TRAINEDDATA}"

TRAIN_LIST="${TRAIN_LIST:-data/train_list_v11.txt}"
EVAL_LIST="${EVAL_LIST:-data/eval_list_v11.txt}"
GOLDEN_MANIFEST="${GOLDEN_MANIFEST:-data/golden/manifest.json}"
BASELINE_REPORT="${BASELINE_REPORT:-reports/baseline_v10.json}"

CKPT_DIR="checkpoints/${MODEL_NAME}"
OUT_MODEL="tessdata/${MODEL_NAME}.traineddata"
LOG_DIR="logs"

mkdir -p "$CKPT_DIR" "$LOG_DIR" base_v10 reports

echo "=== [1/6] CHECKING PREREQUISITES & DATASETS ==="
if [ ! -f "$TRAIN_LIST" ] || [ ! -s "$TRAIN_LIST" ]; then
  echo "[INFO] Generating training list from data/training_manifest_v11.json..."
  python3 scripts/prepare_v11_training.py
fi

if [ ! -f "$TRAIN_LIST" ] || [ ! -s "$TRAIN_LIST" ]; then
  echo "[ERROR] $TRAIN_LIST still missing. Aborting."
  exit 1
fi

echo "[INFO] Training list lines: $(wc -l < "$TRAIN_LIST")"
echo "[INFO] Eval list lines:     $(wc -l < "$EVAL_LIST")"

# Resolve starting checkpoint
if [ -f "$BASE_CKPT" ]; then
  CONTINUE_FROM="$BASE_CKPT"
  echo "[INFO] Resuming from checkpoint: $CONTINUE_FROM"
elif [ -f "$BASE_TRAINEDDATA" ]; then
  echo "[INFO] Extracting LSTM from $BASE_TRAINEDDATA -> base_v10/mya_dhamma_v10.lstm"
  combine_tessdata -e "$BASE_TRAINEDDATA" base_v10/mya_dhamma_v10.lstm
  CONTINUE_FROM="base_v10/mya_dhamma_v10.lstm"
else
  echo "[ERROR] Base model $BASE_TRAINEDDATA not found."
  exit 1
fi

echo "=== [2/6] STAGE 1: BALANCED FINE-TUNING (lr=0.0002, iters=2000) ==="
lstmtraining \
  --continue_from "$CONTINUE_FROM" \
  --traineddata "$STARTER_DATA" \
  --model_output "$CKPT_DIR/${MODEL_NAME}" \
  --train_listfile "$TRAIN_LIST" \
  --eval_listfile "$EVAL_LIST" \
  --learning_rate 0.0002 \
  --max_iterations 2000 \
  --target_error_rate 0.015 \
  --debug_interval 0 2>&1 | tee "${LOG_DIR}/train_${MODEL_NAME}_stage1.log"

echo "=== [3/6] STAGE 2: LR DECAY POLISH (lr=0.00008, iters=3000) ==="
lstmtraining \
  --continue_from "$CKPT_DIR/${MODEL_NAME}_checkpoint" \
  --traineddata "$STARTER_DATA" \
  --model_output "$CKPT_DIR/${MODEL_NAME}" \
  --train_listfile "$TRAIN_LIST" \
  --eval_listfile "$EVAL_LIST" \
  --learning_rate 0.00008 \
  --max_iterations 3000 \
  --target_error_rate 0.012 \
  --debug_interval 0 2>&1 | tee "${LOG_DIR}/train_${MODEL_NAME}_stage2.log"

# Select best candidate checkpoint
BEST_CKPT=$(ls "$CKPT_DIR"/${MODEL_NAME}_*_*.checkpoint 2>/dev/null | sort -t_ -k4 -n | head -n 1 || true)
if [ -z "$BEST_CKPT" ]; then
  BEST_CKPT="$CKPT_DIR/${MODEL_NAME}_checkpoint"
fi
echo "[INFO] Best checkpoint identified: $BEST_CKPT"

echo "=== [4/6] EXPORTING TRAINEDDATA MODEL ==="
lstmtraining \
  --stop_training \
  --continue_from "$BEST_CKPT" \
  --traineddata "$STARTER_DATA" \
  --model_output "$OUT_MODEL"

combine_tessdata -d "$OUT_MODEL"

echo "=== [5/6] RUNNING AUTOMATED GOLDEN-SET REGRESSION GATE ==="
HYP_DIR="reports/hyp_${MODEL_NAME}"
mkdir -p "$HYP_DIR"

python3 - <<PY
import os, json, subprocess
from pathlib import Path
from core.myanmar_normalizer import normalize_myanmar_unicode

manifest = json.loads(Path("${GOLDEN_MANIFEST}").read_text(encoding="utf-8"))
for p in manifest.get("pages", []):
    img = p["image_path"]
    pid = p["page_id"]
    out_txt = Path("${HYP_DIR}") / f"{pid}.txt"
    proc = subprocess.run(
        ["tesseract", img, "stdout", "--tessdata-dir", "tessdata", "-l", "${MODEL_NAME}", "--psm", "6"],
        capture_output=True, text=True, check=True
    )
    norm = normalize_myanmar_unicode(proc.stdout)
    out_txt.write_text(norm, encoding="utf-8")
    print(f"[EVAL-OCR] Wrote {out_txt.name} ({len(norm)} chars)")
PY

GATE_ARGS=()
if [ -f "$BASELINE_REPORT" ]; then
  GATE_ARGS=(--baseline "$BASELINE_REPORT" --max-regression-pp 0.20)
fi

python3 scripts/eval_golden.py \
  --manifest "$GOLDEN_MANIFEST" \
  --hyp-dir "$HYP_DIR" \
  --model "$MODEL_NAME" \
  --psm 6 \
  --save-report "reports/eval_${MODEL_NAME}.json" \
  "${GATE_ARGS[@]}"

echo "=== [6/6] LOCAL BENCHMARK COMPARISON ==="
python3 scripts/benchmark_v10_vs_v11.py || true

echo "============================================================================="
echo "[COMPLETE] Finished ${MODEL_NAME} training and verification!"
echo "Model output: ${OUT_MODEL}"
echo "============================================================================="
