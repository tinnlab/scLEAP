#!/usr/bin/env bash
set -euo pipefail

# --- settings ---
ENV_FILE="experiments/configs/scTab.yml"
ENV_NAME="scTab"
SCRIPT="experiments/cell_type_annotation/scTab/run_scTab.py"
DATA_DIR="./data"
RESULT_DIR="./results"

# --- step 1: create conda env if not exist ---
source "$(conda info --base)/etc/profile.d/conda.sh"

if ! conda env list | grep -q "^${ENV_NAME}"; then
    echo "[INFO] Creating conda environment ${ENV_NAME} from ${ENV_FILE}"
    conda env create -f "${ENV_FILE}" -n "${ENV_NAME}"
else
    echo "[INFO] Conda env ${ENV_NAME} already exists"
fi

# --- step 2: activate env ---
set +u
conda activate "${ENV_NAME}"
set -u

# --- step 3: run scTab ---
echo "[INFO] Running scTab..."
python "${SCRIPT}" \
    --device 1 \
    --data_dir "${DATA_DIR}" \
    --result_dir "${RESULT_DIR}"
