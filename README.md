# scLEAD

`scLEAD` is the research codebase for single-cell language-model-guided embedding, annotation, zero-shot transfer, and clustering analyses.

## Repository Layout

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

`sclead.yml` captures the full research environment. For a lighter editable install:

```bash
pip install -e .
```

Core runtime dependencies are declared in `pyproject.toml` and mirrored in `src/requirements.txt`.

## Main Workflows

Detailed dataset placement and run commands for analysis scripts are documented in [analyses/README.md](/home/dungp/scLEAD/analyses/README.md).

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

## Data

The analysis scripts expect local training and evaluation datasets that are not bundled in this repository. Packaged ontology metadata lives in `src/sclead/ct_descriptions/`.

