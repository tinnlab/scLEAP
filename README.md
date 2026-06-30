# scLEAD

`scLEAD` is a research codebase for single-cell language-model-guided embedding, annotation, zero-shot transfer, and clustering analyses.

The repository contains the Python package code under `src/sclead/` and runnable analysis workflows under `analyses/`.

## Repository Structure

```text
scLEAD/
├─ analyses/
│  ├─ within_tissue_annotation/
│  ├─ zero_shot_annotation/
│  └─ cell_clustering/
│  └─ data/
├─ src/sclead/
│  ├─ model.py
│  ├─ dataset.py
│  ├─ dataset_gpu.py
│  ├─ gpu_parquet_dataset.py
│  ├─ loss.py
│  ├─ utils.py
│  └─ ct_descriptions/
├─ pyproject.toml
├─ sclead.yml
└─ README.md
```

## Installation

The recommended setup is:

1. Create the Conda environment.
2. Install PyTorch using the official PyTorch installation selector.
3. Install RAPIDS using the official RAPIDS installation selector.
4. Install `scLEAD` in editable mode.

PyTorch and RAPIDS are installed separately because their correct wheels depend on the local GPU driver, CUDA version, Python version, and platform.

### 1. Create the Conda environment

From the repository root:

```bash
conda env create -f sclead.yml
conda activate sclead
```

If you prefer to create the environment manually:

```bash
conda create -n sclead python=3.11 -y
conda activate sclead
conda install -c conda-forge pip -y
```

Use `python -m pip` instead of plain `pip` to ensure packages are installed into the active Conda environment:

```bash
which python
python -m pip --version
```

### 2. Install PyTorch

Install PyTorch by following the official PyTorch installation selector:

* Select your operating system.
* Select `pip`.
* Select `Python`.
* Select the CUDA version supported by your system.
* Copy and run the generated command.

Example for a CUDA 13.2 system:

```bash
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu132
```

Verify the installation:

```bash
python - <<'PY'
import torch

print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("torch cuda:", torch.version.cuda)

if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
PY
```

### 3. Install RAPIDS

Install RAPIDS by following the official RAPIDS installation selector.

Use the selector to choose:

* Install method: `pip` or `conda`
* CUDA version: matching your system, for example CUDA 13
* Python version: compatible with RAPIDS
* RAPIDS version: the latest compatible stable release

For pip installs, RAPIDS packages use CUDA-specific names such as `cudf-cu13`, `cuml-cu13`, and `cugraph-cu13`.

Example CUDA 13 pip command:

```bash
python -m pip install \
    --extra-index-url=https://pypi.nvidia.com \
    cudf-cu13 dask-cudf-cu13 cuml-cu13 cugraph-cu13 nx-cugraph-cu13 \
    cucim-cu13 pylibraft-cu13 raft-dask-cu13 cuvs-cu13
```

If the RAPIDS selector gives exact versions, use the exact command from the selector instead of manually editing package versions.

Verify the RAPIDS installation:

```bash
python - <<'PY'
import cudf
import cuml
import cugraph

print("cudf:", cudf.__version__)
print("cuml:", cuml.__version__)
print("cugraph:", cugraph.__version__)
print(cudf.Series([1, 2, 3]))
PY
```

### 4. Install scLEAD

After PyTorch and RAPIDS are installed, install this repository in editable mode:

```bash
python -m pip install -e .
```

Verify that `scLEAD` imports correctly:

```bash
python - <<'PY'
import sclead
print("scLEAD import OK")
PY
```

## Notes About GPU Dependencies

PyTorch and RAPIDS should not be pinned blindly inside `sclead.yml` or `pyproject.toml`, because the correct installation depends on the machine.

Recommended practice:

* Keep general Python dependencies in `sclead.yml` and `pyproject.toml`.
* Install PyTorch from the official PyTorch selector.
* Install RAPIDS from the official RAPIDS selector.
* Use `python -m pip` to avoid accidentally installing packages outside the active Conda environment.

If installation fails with an error such as:

```text
No matching distribution found
```

check:

```bash
python --version
nvidia-smi
python -m pip --version
```

Then regenerate the PyTorch or RAPIDS install command from the official selector using the correct Python and CUDA settings.

## Data and Outputs

The repository does not bundle large training datasets, checkpoints, or generated results. The analysis scripts expect local paths for inputs and outputs.

Packaged ontology metadata lives in:

```text
src/sclead/ct_descriptions/
```

Recommended data layout:

```text
data/
├─ <tissue_name>/
│  ├─ train.h5ad
│  └─ test.h5ad
├─ <parquet_tissue_name>/
│  ├─ train_parquets/
│  ├─ test_parquets/
│  ├─ train_ontology_to_int.json
│  ├─ train_celltype_to_int.json
│  ├─ all_ontology_to_int.json
│  └─ all_celltype_to_int.json
└─ zeroshot-data/
   ├─ GSE111976.h5ad
   └─ ...
```

Detailed run commands and expected file placement for each workflow are documented in:

```text
analyses/README.md
```

## Main Workflows

Each workflow is exposed as a script under `analyses/` and can be inspected with `--help`.

### Within-tissue annotation

```bash
python analyses/within_tissue_annotation/run_scLEAD.py --help
```

### Foundation-model training

```bash
python analyses/zero_shot_annotation/train_foundation_model.py --help
```

### Zero-shot prediction

```bash
python analyses/zero_shot_annotation/run_zero_shot_prediction.py --help
```

### Clustering analysis

```bash
python analyses/cell_clustering/run_clustering_scLEAD_new.py --help
```

## Suggested Starting Points

If you are new to the codebase, use this order:

1. Create and activate the `sclead` environment.
2. Install PyTorch from the official PyTorch selector.
3. Install RAPIDS from the official RAPIDS selector.
4. Install the package with `python -m pip install -e .`.
5. Read `analyses/README.md` for the expected input layout.
6. Run the relevant script with `--help`.
7. Point the script to your local data, checkpoints, and output directory.

## Additional Notes

* The analysis scripts expect local training and evaluation datasets that are not bundled in this repository.
* Scripts are intended to be run from the repository root.
* Large raw data, checkpoints, and generated result folders should stay outside the source tree when possible.
* Use `.gitignore` to exclude large generated outputs, local datasets, logs, and checkpoints.
