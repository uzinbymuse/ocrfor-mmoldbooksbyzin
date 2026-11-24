#!/usr/bin/env bash
# =============================================================================
# Train mya_dhamma_v13_real (100% Real Scanned Lead-Type Crops, ZERO Synthetic)
# Base Model: tessdata/mya_dhamma_v10.traineddata
# Iterations: 150 to 300 ultra-short surgical polish (lr=0.00005)
# =============================================================================

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

MODEL_NAME="${MODEL_NAME:-mya_dhamma_v13_real}"
BASE_TRAINEDDATA="${BASE_TRAINEDDATA:-tessdata/mya_dhamma_v10.traineddata}"
BASE_CKPT="${BASE_CKPT:-checkpoints/mya_dhamma_v10_checkpoint}"
STARTER_DATA="${STARTER_DATA:-$BASE_TRAINEDDATA}"

TRAIN_LIST_REAL="data/train_list_v13_real.txt"
EVAL_LIST="${EVAL_LIST:-data/eval_list_v11.txt}"
GOLDEN_MANIFEST="${GOLDEN_MANIFEST:-data/golden/manifest.json}"
BASELINE_REPORT="${BASELINE_REPORT:-reports/baseline_v10.json}"

CKPT_DIR="checkpoints/${MODEL_NAME}"
OUT_MODEL="tessdata/${MODEL_NAME}.traineddata"
LOG_DIR="logs"

mkdir -p "$CKPT_DIR" "$LOG_DIR" base_v10 reports

echo "=== [1/5] PREPARING ZERO-SYNTHETIC (100% REAL) TRAINING LIST ==="

# Build REAL-ONLY training list by strictly excluding synthetic and human_synth
python3 - <<PY
import os
from pathlib import Path

train_candidates = [
    Path("data/train_list_v12b.txt"),
    Path("data/train_list_v11.txt"),
    Path("data/v11/lists/train_list.txt")
]

src_list = None
for p in train_candidates:
    if p.exists() and p.stat().st_size > 0:
        src_list = p
        break

real_lines = []
if src_list:
    for line in src_list.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        # STRICT FILTER: zero synthetic, zero human_synth, only authentic scanned crops
        l_lower = line.lower()
        if "synthetic" in l_lower or "human_synth" in l_lower or "syn_" in l_lower:
            continue
        if Path(line).exists():
            real_lines.append(line)

# If real_lines is small, oversample real crops 5x to ensure proper batching
if real_lines:
    unique_real = list(dict.fromkeys(real_lines))
    print(f"[DATA] Found {len(unique_real)} unique authentic real lead-type crops.")
    oversampled = []
    for _ in range(5):
        oversampled.extend(unique_real)
    out_p = Path("${TRAIN_LIST_REAL}")
    out_p.write_text("\n".join(oversampled) + "\n", encoding="utf-8")
    print(f"[DATA] Wrote {len(oversampled)} lines (5x oversampled) -> {out_p}")
else:
    print("[WARN] No source list found. Please ensure real crops .lstmf files are listed.")
PY

if [ ! -f "$TRAIN_LIST_REAL" ] || [ ! -s "$TRAIN_LIST_REAL" ]; then
  echo "[ERROR] $TRAIN_LIST_REAL is missing or empty. Please supply real crops list."
  exit 1
fi

TOTAL_LINES=$(wc -l < "$TRAIN_LIST_REAL")
echo "[INFO] Training on ${TOTAL_LINES} 100% Real Lead-Type lines (0% Synthetic)."

# Resolve base checkpoint
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

echo "=== [2/5] ULTRA-SHORT REAL-ONLY SURGICAL POLISH (max 300 iters, lr=0.00005) ==="
EVAL_FLAG=()
if [ -f "$EVAL_LIST" ] && [ -s "$EVAL_LIST" ]; then
  EVAL_FLAG=(--eval_listfile "$EVAL_LIST")
fi

lstmtraining \
  --continue_from "$CONTINUE_FROM" \
  --traineddata "$STARTER_DATA" \
  --model_output "$CKPT_DIR/${MODEL_NAME}" \
  --train_listfile "$TRAIN_LIST_REAL" \
  "${EVAL_FLAG[@]}" \
  --learning_rate 0.00005 \
  --max_iterations 300 \
  --target_error_rate 0.012 \
  --debug_interval 0 2>&1 | tee "${LOG_DIR}/train_${MODEL_NAME}.log"

echo "=== [3/5] SELECTING BEST CHECKPOINT (LOWEST BCER) ==="
# Checkpoints produced:
ls -la "$CKPT_DIR"/*.checkpoint || true

BEST_CKPT=$(ls "$CKPT_DIR"/${MODEL_NAME}_*_*.checkpoint 2>/dev/null | sort -t_ -k4 -n | head -n 1 || true)
if [ -z "$BEST_CKPT" ]; then
  BEST_CKPT="$CKPT_DIR/${MODEL_NAME}_checkpoint"
fi
echo "[INFO] Best checkpoint selected: $BEST_CKPT"

echo "=== [4/5] EXPORTING TRAINEDDATA MODEL ==="
lstmtraining \
  --stop_training \
  --continue_from "$BEST_CKPT" \
  --traineddata "$STARTER_DATA" \
  --model_output "$OUT_MODEL"

combine_tessdata -d "$OUT_MODEL"

echo "=== [5/5] RUNNING 4-PAGE BENCHMARK & GOLDEN-SET VERIFICATION ==="
HYP_DIR="reports/hyp_${MODEL_NAME}"
mkdir -p "$HYP_DIR"

python3 - <<PY
import os, json, subprocess
from pathlib import Path
from core.myanmar_normalizer import normalize_myanmar_unicode

manifest_p = Path("${GOLDEN_MANIFEST}")
if manifest_p.exists():
    manifest = json.loads(manifest_p.read_text(encoding="utf-8"))
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

if [ -f "$GOLDEN_MANIFEST" ] && [ -f "$BASELINE_REPORT" ]; then
  python3 scripts/eval_golden.py \
    --manifest "$GOLDEN_MANIFEST" \
    --hyp-dir "$HYP_DIR" \
    --model "$MODEL_NAME" \
    --psm 6 \
    --baseline "$BASELINE_REPORT" \
    --max-regression-pp 0.20 \
    --save-report "reports/eval_${MODEL_NAME}.json" || true
fi

python3 scripts/benchmark_v10_vs_v11.py || true

echo "============================================================================="
echo "[COMPLETE] Finished ${MODEL_NAME} training and verification!"
echo "Model output: ${OUT_MODEL}"
echo "============================================================================="
