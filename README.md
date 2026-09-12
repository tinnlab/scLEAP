# scLEAP

**scLEAP: single-cell annotation using Expression–Language Alignment and Poincaré geometry**

`scLEAP` is a research codebase for single-cell annotation using expression–language alignment and Poincaré geometry, including zero-shot transfer and clustering analyses.

The repository contains the Python package under `src/scleap/` and runnable analysis workflows under `analyses/`.

## Repository Structure

```text
scLEAP/
├─ analyses/
│  ├─ within_tissue_annotation/
│  ├─ zero_shot_annotation/
│  └─ cell_clustering/
├─ data/
├─ src/scleap/
│  ├─ model.py
│  ├─ dataset.py
│  ├─ dataset_gpu.py
│  ├─ gpu_parquet_dataset.py
│  ├─ loss.py
│  ├─ utils.py
│  └─ ct_descriptions/
├─ pyproject.toml
├─ scleap.yml
└─ README.md
```

## Installation

The recommended setup is to create the Conda environment, install GPU-specific dependencies separately, and then install `scLEAP` in editable mode.

PyTorch and RAPIDS are not pinned directly in `scleap.yml` or `pyproject.toml` because the correct installation command depends on the local GPU driver, CUDA version, Python version, and platform.

### 1. Create the Conda Environment

From the repository root, run:

```bash
conda env create -f scleap.yml
conda activate scleap
```

Alternatively, create the environment manually:

```bash
conda create -n scleap python=3.11 -y
conda activate scleap
conda install -c conda-forge pip -y
```

Use `python -m pip` instead of plain `pip` to ensure packages are installed into the active Conda environment:

```bash
which python
python -m pip --version
```

### 2. Install PyTorch

Install PyTorch using the official PyTorch installation selector:

https://pytorch.org/get-started/locally/

Choose the appropriate operating system, package manager, Python version, and CUDA version for your system, then run the generated command.

After installation, verify that PyTorch can detect the GPU:

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

Install RAPIDS using the official RAPIDS installation guide:

https://docs.rapids.ai/install/

Use the command generated for your CUDA version, Python version, and preferred installation method.

After installation, verify that the main RAPIDS packages import correctly:

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

### 4. Install FAISS
```bash
pip install faiss-gpu
```

### 5. Install scLEAP

After installing PyTorch and RAPIDS, install this repository in editable mode:

```bash
python -m pip install -e .
```

Verify that the package imports correctly:

```bash
python - <<'PY'
import scleap
print("scLEAP import OK")
PY
```

## Notes on GPU Dependencies

PyTorch and RAPIDS should be installed separately from the core Python environment because their package versions depend on the local CUDA and driver configuration.

Recommended practice:

* Keep general Python dependencies in `scleap.yml` and `pyproject.toml`.
* Install PyTorch from the official PyTorch installation selector.
* Install RAPIDS from the official RAPIDS installation guide.
* Use `python -m pip` to avoid installing packages outside the active Conda environment.

If installation fails with an error such as:

```text
No matching distribution found
```

check the Python version, CUDA driver, and pip environment:

```bash
python --version
nvidia-smi
python -m pip --version
```

Then regenerate the PyTorch or RAPIDS installation command using the official installation pages.

## Data and Outputs

Large training datasets, checkpoints, and generated results are not bundled with this repository. The analysis scripts expect local paths for input data, checkpoints, and output directories.

Packaged ontology metadata is included under:

```text
src/scleap/ct_descriptions/
```

The recommended data layout is:

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

Detailed run commands and expected file placement for each workflow are provided in:

```text
analyses/README.md
```

## Main Workflows

Each analysis workflow is available as a runnable script under `analyses/`. Use `--help` to inspect the available arguments.

### Within-Tissue Annotation

```bash
python analyses/within_tissue_annotation/run_scLEAP.py --help
```

### Foundation-Model Training

```bash
python analyses/zero_shot_annotation/train_foundation_model.py --help
```

### Zero-Shot Prediction

```bash
python analyses/zero_shot_annotation/run_zero_shot_prediction.py --help
```

### Cell Clustering

```bash
python analyses/cell_clustering/run_clustering_scLEAP_new.py --help
```

## Suggested Starting Point

For a new setup, use the following order:

1. Create and activate the `scleap` Conda environment.
2. Install PyTorch using the official PyTorch installation selector.
3. Install RAPIDS using the official RAPIDS installation guide.
4. Install this repository with `python -m pip install -e .`.
5. Review `analyses/README.md` for the expected data layout and workflow-specific commands.
6. Run the relevant analysis script with `--help`.
7. Provide local paths to the required data, checkpoints, and output directories.

## Additional Notes

* Analysis scripts are intended to be run from the repository root.
* Large raw datasets, checkpoints, logs, and generated outputs should not be committed to the repository.
* Use `.gitignore` to exclude local datasets, intermediate files, checkpoints, logs, and generated result directories.
* If datasets or outputs are stored outside the repository, pass absolute paths using the appropriate command-line arguments instead of editing the scripts.
