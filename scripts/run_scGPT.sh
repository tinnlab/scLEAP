#!/usr/bin/env bash
set -euo pipefail

# --- settings ---
ENV_FILE="experiments/configs/scGPT.yml"   # your conda YAML
# If your YAML has a different env name, it will be auto-detected below
DEFAULT_ENV_NAME="scGPT"

MODEL_DIR="experiments/scGPT/model_weight"
mkdir -p "$MODEL_DIR"
# Google Drive folder with all 3 weight files
GDRIVE_FOLDER_URL="https://drive.google.com/drive/folders/1oWh_-ZRdhtoGQ2Fw24HP41FgLoomVo-y?usp=drive_link"

# --- conda setup (make 'conda activate' work in scripts) ---
if ! command -v conda >/dev/null 2>&1; then
  echo "[ERROR] conda not found in PATH." >&2
  exit 1
fi
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"

# Avoid MKL hook issues with set -u
export MKL_INTERFACE_LAYER=${MKL_INTERFACE_LAYER:-LP64}
export CONDA_MKL_INTERFACE_LAYER_BACKUP=${CONDA_MKL_INTERFACE_LAYER_BACKUP:-}

# --- detect env name from YAML (fallback to DEFAULT_ENV_NAME) ---
if [ -f "$ENV_FILE" ]; then
  YAML_ENV_NAME=$(awk -F': *' '/^name:/ {print $2; exit}' "$ENV_FILE" || true)
  ENV_NAME="${YAML_ENV_NAME:-$DEFAULT_ENV_NAME}"
else
  echo "[ERROR] Env file not found: $ENV_FILE" >&2
  exit 1
fi
echo "[INFO] Using conda env name: $ENV_NAME"

# --- create env if missing ---
if ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  echo "[INFO] Creating conda environment from $ENV_FILE"
  set +u; conda env create -f "$ENV_FILE"; set -u
else
  echo "[INFO] Conda env '$ENV_NAME' already exists"
fi

# --- activate env ---
set +u; conda activate "$ENV_NAME"; set -u

# --- install flash-attention (AFTER torch is installed by the YAML) ---
# NOTE: 1.0.4 is very old and often requires building from source.
# If build fails, you may need a newer torch/CUDA stack and a newer flash-attn.
pip install flash-attn==1.0.4

# --- ensure gdown is available for Google Drive downloads ---
python - <<'PY'
import importlib.util, sys
spec = importlib.util.find_spec("gdown")
sys.exit(0 if spec else 1)
PY
if [ $? -ne 0 ]; then
  pip install gdown
fi

# --- download model weights (grab the whole folder for convenience) ---
mkdir -p "$MODEL_DIR"
echo "[INFO] Downloading model weights into $MODEL_DIR"
# gdown --folder handles the three files in one shot
gdown --folder "$GDRIVE_FOLDER_URL" -O "$MODEL_DIR" --remaining-ok

# If you prefer to download specific files by ID instead of the folder, uncomment:
# gdown "14AebJfGOUF047Eg40hk57HCtrb0fyDTm" -O "$MODEL_DIR"
# gdown "1hh2zGKyWAx3DyovD30GStZ3QlzmSqdk1" -O "$MODEL_DIR"
# gdown "1H3E_MJ-Dl36AQV6jLbna2EdvgPaqvqcC" -O "$MODEL_DIR"

# --- run scGPT ---
echo "[INFO] Running scGPT..."
python experiments/cell_type_annotation/scGPT/run_scGPT.py \
  --device 1 \
  --data_dir "./data/" \
  --result_dir "./results"
