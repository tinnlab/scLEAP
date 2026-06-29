# scLEAD

`scLEAD` is the research codebase for single-cell language-model-guided embedding, annotation, zero-shot transfer, and clustering analyses.

This repository contains the package code under `src/sclead/` and the runnable analysis workflows under `analyses/`. If you want the shortest path to a working setup, start with the install section below, then follow the workflow that matches your task.

## What's In This Repo

```text
scLEAD/
├─ analyses/
│  ├─ within_tissue_annotation/
│  ├─ zero_shot_annotation/
│  └─ cell_clustering/
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

`sclead.yml` captures the full research environment used for the paper-style analyses.

For local development or a lighter editable install:

```bash
pip install -e .
```

Core runtime dependencies are declared in `pyproject.toml` and mirrored in `src/requirements.txt`.

## Data And Outputs

The repository does not bundle the large training datasets or generated results. The analysis scripts expect you to provide local paths for inputs and outputs.

The packaged ontology metadata lives in `src/sclead/ct_descriptions/` and is installed with the Python package.

Recommended layout:

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

The detailed run commands and expected file placement for each workflow are documented in [analyses/README.md](analyses/README.md).

## Main Workflows

Each workflow is exposed as a script under `analyses/` and can be inspected with `--help`.

Within-tissue annotation:

```bash
python analyses/within_tissue_annotation/run_scLEAD.py --help
```

Foundation-model training:

```bash
python analyses/zero_shot_annotation/train_foundation_model.py --help
```

Zero-shot prediction:

```bash
python analyses/zero_shot_annotation/run_zero_shot_prediction.py --help
```

Clustering analysis:

```bash
python analyses/cell_clustering/run_clustering_scLEAD_new.py --help
```

## Suggested Starting Points

If you are new to the codebase, use this order:

1. Install the package with `pip install -e .`.
2. Read [analyses/README.md](analyses/README.md) for the exact input layout.
3. Run the relevant script with `--help` to check available options.
4. Point the script to your local data, checkpoints, and output directory.

## Notes

- The analysis scripts expect local training and evaluation datasets that are not bundled in this repository.
- The scripts are intended to be run from the repository root, but they resolve repo-relative paths so you do not need to hard-code a specific working directory.
- Large raw data, checkpoints, and generated result folders should stay outside the source tree when possible.

