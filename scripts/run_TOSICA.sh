#!/usr/bin/env bash
set -euo pipefail

# --- settings ---
REPO_URL="https://github.com/JackieHanLab/TOSICA.git"   # <-- replace with official TOSICA repo URL
CLONE_DIR="./experiments/cell_type_annotation/TOSICA/TOSICA_repo"
SCRIPT_SRC="./experiments/cell_type_annotation/TOSICA/run_TOSICA.py"
SCRIPT_DST="$CLONE_DIR/TOSICA/run_TOSICA.py"
ENV_FILE="./experiments/configs/TOSICA.yml"
ENV_NAME="TOSICA"

mkdir -p "$(dirname "$CLONE_DIR")"

# --- step 1: clone the repo if needed ---
if [ ! -d "$CLONE_DIR/.git" ]; then
    echo "[INFO] Cloning TOSICA repo into $CLONE_DIR"
    git clone "$REPO_URL" "$CLONE_DIR"
else
    echo "[INFO] TOSICA repo already exists in $CLONE_DIR"
fi

# --- step 2: create conda env from lock file (only if not already present) ---
source "$(conda info --base)/etc/profile.d/conda.sh"
if ! conda env list | grep -q "^$ENV_NAME"; then
    echo "[INFO] Creating conda environment $ENV_NAME from $ENV_FILE"
    conda env create -f "$ENV_FILE"
else
    echo "[INFO] Conda env $ENV_NAME already exists"
fi

set +u
conda activate "$ENV_NAME"
set -u



# --- step 3: copy your run script into the repo ---
echo "[INFO] Copying $SCRIPT_SRC -> $SCRIPT_DST"
cp "$SCRIPT_SRC" "$SCRIPT_DST"

pip install "$CLONE_DIR"  # Install the TOSICA package
set +u
conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=11.1 -c pytorch -c conda-forge -y
set -u

# --- step 4: run TOSICA ---
echo "[INFO] Running TOSICA..."
python "$SCRIPT_DST" \
    --gpu 1 \
    --data_dir "./data" \
    --result_dir "./results"
