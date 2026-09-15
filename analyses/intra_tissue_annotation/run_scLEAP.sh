#!/usr/bin/env bash
set -Eeuo pipefail

[[ -n "${BASH_VERSION:-}" ]] || { echo "Run with Bash: bash $0" >&2; exit 1; }
command -v nvidia-smi >/dev/null 2>&1 || { echo "nvidia-smi not found." >&2; exit 1; }

# -------------------- Conda --------------------
if ! command -v conda &>/dev/null; then
  echo "ERROR: conda not found in PATH." >&2
  echo "Try: source ~/miniconda3/etc/profile.d/conda.sh" >&2
  exit 1
fi
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
echo "Activating conda environment: scleap"
conda activate scleap

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# -------------------- Config --------------------
CSV_PATH="${CSV_PATH:-${REPO_ROOT}/data/tissue_cell_counts.csv}"
PYTHON_BIN="${PYTHON_BIN:-python}"
SCRIPT_PATH="${SCRIPT_PATH:-${SCRIPT_DIR}/run_scLEAP.py}"

DATA_DIR="${DATA_DIR:-${REPO_ROOT}/data}"
SAVE_ROOT="${SAVE_ROOT:-${REPO_ROOT}/outputs/intra_tissue_annotation}"

GPU_ID="${GPU_ID:-4}"
MAX_EPOCH="${MAX_EPOCH:-200}"

M1="${M1:-1.0}"
M2="${M2:-0.25}"
M3="${M3:-0.1}"
M4="${M4:-0.8}"

HIDDEN_DIM="${HIDDEN_DIM:-768}"
S="${S:-30}"

L_CT="${L_CT:-1.0}"
L_TT="${L_TT:-0.1}"
L_CONTRASTIVE="${L_CONTRASTIVE:-0.1}"
L_GRAPH="${L_GRAPH:-0.1}"

USE_MARKER="${USE_MARKER:-1}"
MARKER_NAME="${MARKER_NAME:-_DONE}"
WRITE_FAILED_MARKER="${WRITE_FAILED_MARKER:-1}"
SKIP_IF_FAILED_EXISTS="${SKIP_IF_FAILED_EXISTS:-0}"

ONLY_TISSUES="${ONLY_TISSUES:-}"

# CPU thread hygiene
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-4}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTHONUNBUFFERED=1

command -v "${PYTHON_BIN}" >/dev/null 2>&1 || { echo "python not found: ${PYTHON_BIN}" >&2; exit 1; }
[[ -f "$CSV_PATH" ]] || { echo "CSV not found: $CSV_PATH" >&2; exit 1; }
[[ -d "$DATA_DIR" ]] || { echo "DATA_DIR not found: $DATA_DIR" >&2; exit 1; }
[[ -f "$SCRIPT_PATH" ]] || { echo "SCRIPT_PATH not found: $SCRIPT_PATH" >&2; exit 1; }

EXP_SAVE_ROOT="${SAVE_ROOT}/scLEAP_full_hidden${HIDDEN_DIM}_s${S}_m${M1}_${M2}_${M3}_${M4}_lct${L_CT}_ltt${L_TT}_lcon${L_CONTRASTIVE}_lgraph${L_GRAPH}"
LOG_ROOT="${EXP_SAVE_ROOT}/logs"
mkdir -p "$LOG_ROOT"

echo "============================================================"
echo "Sequential tissue runner (Parquet only)"
echo "CSV_PATH      : $CSV_PATH"
echo "DATA_DIR      : $DATA_DIR"
echo "SCRIPT_PATH   : $SCRIPT_PATH"
echo "SAVE_ROOT     : $EXP_SAVE_ROOT"
echo "GPU_ID        : $GPU_ID"
echo "MAX_EPOCH     : $MAX_EPOCH"
echo "============================================================"

# -------------------- Helpers --------------------
trim() {
  local x="$1"
  x="${x#"${x%%[![:space:]]*}"}"
  x="${x%"${x##*[![:space:]]}"}"
  x="${x%\"}"
  x="${x#\"}"
  printf '%s' "$x"
}

in_csv_list() {
  local item="$1"
  local csv="$2"
  [[ -z "$csv" ]] && return 0
  IFS=',' read -r -a arr <<< "$csv"
  local e
  for e in "${arr[@]}"; do
    e="$(trim "$e")"
    [[ "$item" == "$e" ]] && return 0
  done
  return 1
}

run_one_tissue() {
  local tissue="$1"
  local tissue_dir="$DATA_DIR/$tissue"
  local out_dir="$EXP_SAVE_ROOT/$tissue"
  local log_file="$LOG_ROOT/${tissue}.log"
  local failed_marker="$out_dir/_FAILED"
  local done_marker="$out_dir/$MARKER_NAME"

  mkdir -p "$out_dir"

  if [[ "$USE_MARKER" == "1" && -f "$done_marker" ]]; then
    echo "[SKIP] $tissue : DONE marker exists"
    return 0
  fi

  if [[ "$SKIP_IF_FAILED_EXISTS" == "1" && -f "$failed_marker" ]]; then
    echo "[SKIP] $tissue : FAILED marker exists"
    return 0
  fi

  if [[ ! -d "$tissue_dir" ]]; then
    echo "[WARN] $tissue : missing parquet directory: $tissue_dir"
    return 0
  fi

  echo
  echo "------------------------------------------------------------"
  echo "[RUN] Tissue: $tissue"
  echo "      Data : $tissue_dir"
  echo "      Log  : $log_file"
  echo "------------------------------------------------------------"

  rm -f "$failed_marker"

  start_ts="$(date +%s)"

  set +e
  CUDA_VISIBLE_DEVICES="$GPU_ID" \
  "$PYTHON_BIN" "$SCRIPT_PATH" \
    --data-dir "$DATA_DIR" \
    --save-root "$EXP_SAVE_ROOT" \
    --tissues "$tissue" \
    --cuda-visible "$GPU_ID" \
    --gpu-id 0 \
    --use-parquet \
    --max-epoch "$MAX_EPOCH" \
    --hidden-dim "$HIDDEN_DIM" \
    --s "$S" \
    --m1 "$M1" \
    --m2 "$M2" \
    --m3 "$M3" \
    --m4 "$M4" \
    --l-ct "$L_CT" \
    --l-tt "$L_TT" \
    --l-contrastive "$L_CONTRASTIVE" \
    --l-graph "$L_GRAPH" \
    >"$log_file" 2>&1
  status=$?
  set -e

  end_ts="$(date +%s)"
  elapsed=$(( end_ts - start_ts ))

  {
    echo
    echo "Time: ${elapsed} seconds"
    echo "Exit status: ${status}"
  } >> "$log_file"

  if [[ $status -eq 0 ]]; then
    if [[ "$USE_MARKER" == "1" ]]; then
      : > "$done_marker"
    fi
    rm -f "$failed_marker"
    echo "[OK] $tissue finished. Time: ${elapsed} seconds"
  else
    if [[ "$WRITE_FAILED_MARKER" == "1" ]]; then
      : > "$failed_marker"
    fi
    echo "[FAIL] $tissue failed with status ${status}. Time: ${elapsed} seconds"
  fi

  return 0
}

# -------------------- Build tissue list from CSV (NO FILTERING) --------------------
mapfile -t TISSUES < <(
  awk -F',' '
    NR==1 { next }
    {
      t = $1
      gsub(/^[ \t"]+|[ \t"]+$/, "", t)
      print t
    }
  ' "$CSV_PATH"
)

if [[ ${#TISSUES[@]} -eq 0 ]]; then
  echo "No tissues found in CSV."
  exit 0
fi

echo "Found ${#TISSUES[@]} tissues from CSV."

# -------------------- Run sequentially --------------------
count_total=0
count_run=0

for tissue in "${TISSUES[@]}"; do
  tissue="$(trim "$tissue")"
  [[ -z "$tissue" ]] && continue

  if ! in_csv_list "$tissue" "$ONLY_TISSUES"; then
    continue
  fi

  count_total=$((count_total + 1))
  run_one_tissue "$tissue"
  count_run=$((count_run + 1))
done

echo
echo "Finished sequential run."
echo "Matched tissues processed: $count_run"
