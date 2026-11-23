#!/usr/bin/env bash
# =============================================================================
# train_mya_dhamma.sh — 3-Stage Curriculum + Golden-Set Checkpoint Selection
# =============================================================================
# Upgraded Tesseract 5 LSTM training pipeline for uzinbymuse/ocrfor-mmoldbooksbyzin.
#
# Key improvements over v1:
#   1. Direct `lstmtraining` (NO tesstrain — avoids Myanmar virama bug).
#   2. Single-line `--psm 7` for `.lstmf` generation (matches cropped line GT).
#   3. 3-Stage Curriculum (Beatson + Dhamma best practice):
#        Stage 1: Synthetic degraded Dhamma/Pali lines   (lr=0.0010)
#        Stage 2: Uncleaned real Burmese lines           (lr=0.0005)
#        Stage 3: Human-verified cleaned Dhamma lines    (lr=0.0002)
#   4. Page/book-grouped train/eval split (prevents same-page data leakage).
#   5. Automatic Golden-Set Checkpoint Selection via `scripts/eval_golden.py`:
#      converts saved Stage-3 checkpoints to `.traineddata`, scores every
#      candidate on the frozen held-out Golden Set, and promotes ONLY the
#      lowest-CER model that passes the per-stratum regression gate.
#
# Usage:
#   MODEL_NAME=mya_dhamma_v10 \
#   BASE_MODEL=tessdata/mya_dhamma_v9_balanced.traineddata \
#   GOLDEN_MANIFEST=data/golden/manifest.json \
#   BASELINE_REPORT=reports/baseline_v9.json \
#   bash scripts/train_mya_dhamma.sh
# =============================================================================

set -euo pipefail

# -----------------------------------------------------------------------------
# Configurable Paths & Hyperparameters
# -----------------------------------------------------------------------------
MODEL_NAME="${MODEL_NAME:-mya_dhamma_v10}"
BASE_MODEL="${BASE_MODEL:-tessdata/mya_dhamma_v9_balanced.traineddata}"
TESSDATA_DIR="${TESSDATA_DIR:-tessdata}"
WORK_DIR="${WORK_DIR:-work/${MODEL_NAME}}"

# 3-Stage Curriculum Directories (each contains paired *.png + *.gt.txt)
STAGE1_SYNTH_DIR="${STAGE1_SYNTH_DIR:-data/stage1_synthetic}"
STAGE2_UNCLEANED_DIR="${STAGE2_UNCLEANED_DIR:-data/stage2_uncleaned}"
STAGE3_CLEANED_DIR="${STAGE3_CLEANED_DIR:-data/ground_truth}"

# Training Hyperparameters
TRAIN_PSM="${TRAIN_PSM:-7}"
JOBS="${JOBS:-$(nproc 2>/dev/null || echo 4)}"
STAGE1_MAX_ITERS="${STAGE1_MAX_ITERS:-3000}"
STAGE2_MAX_ITERS="${STAGE2_MAX_ITERS:-2000}"
STAGE3_MAX_ITERS="${STAGE3_MAX_ITERS:-1500}"
STAGE1_LR="${STAGE1_LR:-0.001}"
STAGE2_LR="${STAGE2_LR:-0.0005}"
STAGE3_LR="${STAGE3_LR:-0.0002}"
APPEND_TOP_LAYER="${APPEND_TOP_LAYER:-false}"  # set true only when expanding unicharset classes

# Golden-Set Checkpoint Selection & Regression Gate
GOLDEN_MANIFEST="${GOLDEN_MANIFEST:-data/golden/manifest.json}"
BASELINE_REPORT="${BASELINE_REPORT:-reports/baseline_v9.json}"
MAX_REGRESSION_PP="${MAX_REGRESSION_PP:-0.25}"
TOP_K_CHECKPOINTS="${TOP_K_CHECKPOINTS:-5}"
EVAL_PSM="${EVAL_PSM:-6}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

mkdir -p "${WORK_DIR}"/{starter,lstmf,checkpoints,candidates,eval_cache}

log() {
  printf '\n[%s] %s\n' "$(date '+%H:%M:%S')" "$*"
}

# -----------------------------------------------------------------------------
# Helper 1: Build .lstmf files for a GT directory using --psm 7
# -----------------------------------------------------------------------------
build_lstmf_stage() {
  local stage_name="$1"
  local gt_dir="$2"
  local out_dir="${WORK_DIR}/lstmf/${stage_name}"
  local list_all="${out_dir}/all.txt"
  local list_train="${out_dir}/train.txt"
  local list_eval="${out_dir}/eval.txt"

  mkdir -p "${out_dir}"
  if [[ ! -d "${gt_dir}" ]]; then
    echo "0"
    return 0
  fi

  local png_count
  png_count=$(find "${gt_dir}" -maxdepth 2 -type f -name '*.png' | wc -l | tr -d ' ')
  if [[ "${png_count}" -eq 0 ]]; then
    echo "0"
    return 0
  fi

  log "Stage [${stage_name}]: Generating .lstmf files for ${png_count} lines (PSM=${TRAIN_PSM}, jobs=${JOBS})..."

  find "${gt_dir}" -maxdepth 2 -type f -name '*.png' | sort | \
    xargs -P "${JOBS}" -I {} bash -c '
      png="$1"
      base="${png%.png}"
      gt="${base}.gt.txt"
      if [[ ! -f "${gt}" ]]; then
        exit 0
      fi
      if [[ ! -f "${base}.lstmf" ]]; then
        OMP_THREAD_LIMIT=1 tesseract "${png}" "${base}" \
          --tessdata-dir "'"${TESSDATA_DIR}"'" \
          -l mya --psm "'"${TRAIN_PSM}"'" lstm.train >/dev/null 2>&1 || true
      fi
    ' _ {}

  find "${gt_dir}" -maxdepth 2 -type f -name '*.lstmf' | sort > "${list_all}"
  local total_lstmf
  total_lstmf=$(wc -l < "${list_all}" | tr -d ' ')
  if [[ "${total_lstmf}" -eq 0 ]]; then
    echo "0"
    return 0
  fi

  # Page/book-grouped 90/10 split (groups by prefix before _lXXX so lines from
  # the same page never leak across train and eval sets)
  "${PYTHON_BIN}" - "${list_all}" "${list_train}" "${list_eval}" <<'PY'
import hashlib, re, sys
from pathlib import Path

all_paths = [p.strip() for p in Path(sys.argv[1]).read_text(encoding="utf-8").splitlines() if p.strip()]
train_out, eval_out = [], []

for p in all_paths:
    stem = Path(p).stem
    # Group key: strip trailing _l001 line suffix if present
    group_key = re.sub(r"_l\d+$", "", stem)
    bucket = int(hashlib.md5(group_key.encode("utf-8")).hexdigest(), 16) % 10
    if bucket == 0:
        eval_out.append(p)
    else:
        train_out.append(p)

# Guarantee at least 1 eval item if dataset is tiny
if not eval_out and len(train_out) > 1:
    eval_out.append(train_out.pop())

Path(sys.argv[2]).write_text("\n".join(train_out) + "\n", encoding="utf-8")
Path(sys.argv[3]).write_text("\n".join(eval_out) + "\n", encoding="utf-8")
PY

  echo "${total_lstmf}"
}

# -----------------------------------------------------------------------------
# Helper 2: Prepare starter traineddata & base .lstm
# -----------------------------------------------------------------------------
prepare_starter() {
  log "Extracting base LSTM & unicharset from ${BASE_MODEL}..."
  combine_tessdata -e "${BASE_MODEL}" "${WORK_DIR}/starter/base.lstm"
  combine_tessdata -u "${BASE_MODEL}" "${WORK_DIR}/starter/base."

  if [[ "${APPEND_TOP_LAYER}" == "true" ]]; then
    log "Expanding unicharset with --norm_mode 2 + --pass_through_recoder..."
    find "${STAGE1_SYNTH_DIR}" "${STAGE2_UNCLEANED_DIR}" "${STAGE3_CLEANED_DIR}" \
      -type f -name '*.gt.txt' 2>/dev/null -exec cat {} + > "${WORK_DIR}/starter/all_gt.txt"

    unicharset_extractor \
      --output_unicharset "${WORK_DIR}/starter/new.unicharset" \
      --norm_mode 2 \
      "${WORK_DIR}/starter/all_gt.txt"

    merge_unicharsets \
      "${WORK_DIR}/starter/base.unicharset" \
      "${WORK_DIR}/starter/new.unicharset" \
      "${WORK_DIR}/starter/merged.unicharset"

    combine_lang_model \
      --input_unicharset "${WORK_DIR}/starter/merged.unicharset" \
      --script_dir "${TESSDATA_DIR}" \
      --output_dir "${WORK_DIR}/starter" \
      --lang "${MODEL_NAME}" \
      --pass_through_recoder

    STARTER_TRAINEDDATA="${WORK_DIR}/starter/${MODEL_NAME}/${MODEL_NAME}.traineddata"
    NUM_CLASSES=$(head -n 1 "${WORK_DIR}/starter/merged.unicharset")
    EXTRA_NET_ARGS=(--append_index 5 --net_spec "[Lfx192 O1c${NUM_CLASSES}]")
  else
    # Continue directly in the existing model's unicharset/recoder space
    # (prevents silent recoder code-space mismatch during fine-tuning)
    STARTER_TRAINEDDATA="${BASE_MODEL}"
    EXTRA_NET_ARGS=()
  fi
}

# -----------------------------------------------------------------------------
# Helper 3: Run one curriculum stage with lstmtraining
# -----------------------------------------------------------------------------
run_curriculum_stage() {
  local stage_name="$1"
  local init_checkpoint="$2"
  local lr="$3"
  local max_iters="$4"
  local extra_flags="$5"

  local train_list="${WORK_DIR}/lstmf/${stage_name}/train.txt"
  local eval_list="${WORK_DIR}/lstmf/${stage_name}/eval.txt"
  local ckpt_dir="${WORK_DIR}/checkpoints/${stage_name}"
  mkdir -p "${ckpt_dir}"

  log "Training Stage [${stage_name}] (lr=${lr}, max_iters=${max_iters})..."

  # shellcheck disable=SC2086
  OMP_THREAD_LIMIT=1 lstmtraining \
    --traineddata "${STARTER_TRAINEDDATA}" \
    --continue_from "${init_checkpoint}" \
    --old_traineddata "${BASE_MODEL}" \
    --train_listfile "${train_list}" \
    --eval_listfile "${eval_list}" \
    --model_output "${ckpt_dir}/${MODEL_NAME}" \
    --learning_rate "${lr}" \
    --max_iterations "${max_iters}" \
    --debug_interval 0 \
    ${extra_flags} \
    "${EXTRA_NET_ARGS[@]}"

  # Return the best checkpoint produced by this stage (fallback to latest)
  if [[ -f "${ckpt_dir}/${MODEL_NAME}_checkpoint" ]]; then
    echo "${ckpt_dir}/${MODEL_NAME}_checkpoint"
  else
    ls -t "${ckpt_dir}"/*.checkpoint | head -n 1
  fi
}

# -----------------------------------------------------------------------------
# Helper 4: Golden-Set Checkpoint Selection & Regression Gate
# -----------------------------------------------------------------------------
select_best_checkpoint_on_golden() {
  local ckpt_dir="$1"

  if [[ ! -f "${GOLDEN_MANIFEST}" ]]; then
    log "WARNING: Golden manifest (${GOLDEN_MANIFEST}) not found; exporting latest checkpoint directly."
    local latest_ckpt="${ckpt_dir}/${MODEL_NAME}_checkpoint"
    lstmtraining \
      --stop_training \
      --continue_from "${latest_ckpt}" \
      --traineddata "${STARTER_TRAINEDDATA}" \
      --model_output "${TESSDATA_DIR}/${MODEL_NAME}.traineddata"
    return 0
  fi

  log "Selecting best checkpoint on held-out Golden Set (${GOLDEN_MANIFEST})..."

  mapfile -t ckpts < <(find "${ckpt_dir}" -maxdepth 1 -type f -name '*.checkpoint' | sort -V | tail -n "${TOP_K_CHECKPOINTS}")
  if [[ -f "${ckpt_dir}/${MODEL_NAME}_checkpoint" ]]; then
    ckpts+=("${ckpt_dir}/${MODEL_NAME}_checkpoint")
  fi

  local best_model=""
  local best_cer="999.0"

  for ckpt in "${ckpts[@]}"; do
    local tag
    tag=$(basename "${ckpt}" | tr '.' '_')
    local cand_model="${WORK_DIR}/candidates/${tag}.traineddata"
    local hyp_dir="${WORK_DIR}/eval_cache/${tag}"
    local rep_json="${WORK_DIR}/candidates/${tag}_report.json"

    mkdir -p "${hyp_dir}"

    lstmtraining \
      --stop_training \
      --continue_from "${ckpt}" \
      --traineddata "${STARTER_TRAINEDDATA}" \
      --model_output "${cand_model}" >/dev/null 2>&1

    # Run candidate model on every page in Golden Set manifest
    "${PYTHON_BIN}" - "${GOLDEN_MANIFEST}" "${WORK_DIR}/candidates" "${tag}" "${EVAL_PSM}" "${hyp_dir}" <<'PY'
import json, subprocess, sys
from pathlib import Path

manifest = Path(sys.argv[1])
tess_dir = sys.argv[2]
lang = sys.argv[3]
psm = sys.argv[4]
hyp_dir = Path(sys.argv[5])
base_dir = manifest.parent

pages = json.loads(manifest.read_text(encoding="utf-8")).get("pages", [])
for p in pages:
    img_path = Path(p["image_path"])
    if not img_path.is_absolute():
        img_path = base_dir / img_path
    out_txt = hyp_dir / f"{p['page_id']}.txt"
    proc = subprocess.run(
        ["tesseract", str(img_path), "stdout", "--tessdata-dir", tess_dir, "-l", lang, "--psm", psm],
        capture_output=True, text=True, check=True,
    )
    out_txt.write_text(proc.stdout, encoding="utf-8")
PY

    local gate_args=()
    if [[ -f "${BASELINE_REPORT}" ]]; then
      gate_args=(--baseline "${BASELINE_REPORT}" --max-regression-pp "${MAX_REGRESSION_PP}")
    fi

    if "${PYTHON_BIN}" scripts/eval_golden.py \
      --manifest "${GOLDEN_MANIFEST}" \
      --hyp-dir "${hyp_dir}" \
      --model "${tag}" \
      --psm "${EVAL_PSM}" \
      --save-report "${rep_json}" \
      "${gate_args[@]}"; then

      local curr_cer
      curr_cer=$("${PYTHON_BIN}" -c "import json; print(json.load(open('${rep_json}'))['overall']['cer_pct'])")
      log "Candidate [${tag}] PASSED gate with Golden CER=${curr_cer}%"

      if "${PYTHON_BIN}" -c "import sys; sys.exit(0 if float('${curr_cer}') < float('${best_cer}') else 1)"; then
        best_cer="${curr_cer}"
        best_model="${cand_model}"
      fi
    else
      log "Candidate [${tag}] FAILED per-stratum regression gate — skipped."
    fi
  done

  if [[ -z "${best_model}" ]]; then
    log "ERROR: No candidate checkpoint passed the Golden Set regression gate!"
    exit 2
  fi

  cp "${best_model}" "${TESSDATA_DIR}/${MODEL_NAME}.traineddata"
  log "PROMOTED winning model -> ${TESSDATA_DIR}/${MODEL_NAME}.traineddata (Golden CER=${best_cer}%)"
}

# -----------------------------------------------------------------------------
# Main Execution
# -----------------------------------------------------------------------------
prepare_starter

CURRENT_CKPT="${WORK_DIR}/starter/base.lstm"

# Stage 1: Synthetic Degraded Dhamma/Pali Warmup
S1_COUNT=$(build_lstmf_stage "stage1_synthetic" "${STAGE1_SYNTH_DIR}")
if [[ "${S1_COUNT}" -gt 0 ]]; then
  CURRENT_CKPT=$(run_curriculum_stage "stage1_synthetic" "${CURRENT_CKPT}" "${STAGE1_LR}" "${STAGE1_MAX_ITERS}" "--randomly_rotate true")
  # Clear append-layer args after first stage so Stage 2/3 continue smoothly
  EXTRA_NET_ARGS=()
else
  log "Skipping Stage 1 (no synthetic lines found in ${STAGE1_SYNTH_DIR})."
fi

# Stage 2: Uncleaned Real Burmese Corpus (e.g., Beatson uncleaned)
S2_COUNT=$(build_lstmf_stage "stage2_uncleaned" "${STAGE2_UNCLEANED_DIR}")
if [[ "${S2_COUNT}" -gt 0 ]]; then
  CURRENT_CKPT=$(run_curriculum_stage "stage2_uncleaned" "${CURRENT_CKPT}" "${STAGE2_LR}" "${STAGE2_MAX_ITERS}" "")
  EXTRA_NET_ARGS=()
else
  log "Skipping Stage 2 (no uncleaned lines found in ${STAGE2_UNCLEANED_DIR})."
fi

# Stage 3: Real Human-Verified Cleaned Dhamma + Beatson Cleaned Lines
S3_COUNT=$(build_lstmf_stage "stage3_cleaned" "${STAGE3_CLEANED_DIR}")
if [[ "${S3_COUNT}" -eq 0 ]]; then
  log "ERROR: Stage 3 cleaned ground truth directory (${STAGE3_CLEANED_DIR}) has 0 .lstmf files."
  exit 1
fi
CURRENT_CKPT=$(run_curriculum_stage "stage3_cleaned" "${CURRENT_CKPT}" "${STAGE3_LR}" "${STAGE3_MAX_ITERS}" "")

# Final Selection on Held-Out Golden Set
select_best_checkpoint_on_golden "${WORK_DIR}/checkpoints/stage3_cleaned"