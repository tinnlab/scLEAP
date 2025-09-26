#!/usr/bin/env bash
set -euo pipefail

# --- settings ---
ENV_FILE="experiments/configs/scLMCT.yml"
ENV_NAME="scLMCT"
SCRIPT="experiments/cell_type_annotation/scLMCT/run_scLMCT.py"
DATA_DIR="./data"
RESULT_DIR="./results"
DEVICE="${DEVICE:-1}"   # override with: DEVICE=0 ./run_scLMCT.sh

# --- sanity checks ---
if ! command -v conda >/dev/null 2>&1; then
  echo "[ERROR] conda not found in PATH. Please install/miniconda and try again." >&2
  exit 1
fi

# --- step 0: prepare conda shim ---
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"

# --- step 1: create conda env if not exist ---
if ! conda env list | awk '{print $1}' | grep -xq "${ENV_NAME}"; then
  echo "[INFO] Creating conda environment ${ENV_NAME} from ${ENV_FILE}"
  conda env create -f "${ENV_FILE}" -n "${ENV_NAME}"
else
  echo "[INFO] Conda env ${ENV_NAME} already exists"
fi

# --- step 2: activate env ---
set +u
conda activate "${ENV_NAME}"
set -u

# --- step 3: ensure output dir ---
mkdir -p "${RESULT_DIR}"

# --- step 4: run scLMCT ---
echo "[INFO] Running scLMCT..."
python "${SCRIPT}" \
  --device "${DEVICE}" \
  --data_dir "${DATA_DIR}" \
  --result_dir "${RESULT_DIR}" \
  "$@"

echo "[INFO] Done."
