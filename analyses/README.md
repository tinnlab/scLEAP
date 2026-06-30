# Analyses Guide

This directory contains runnable workflows for reproducing the main analyses. The scripts resolve paths relative to the repository, so they can be executed from the repository root without depending on a specific working directory.

## Data Availability and Reproducibility

Data, results, and checkpoints are available through the following Seafile link:

[Download data, results, and checkpoints](https://seafile.tinnguyen-lab.com/d/4276c944e5934d30bd4b/)

## Data Layout

Place all required data under the [`data`](../data) directory using the following structure:

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

## Within-Tissue Annotation

### Input

The within-tissue annotation workflow supports both H5AD and Parquet inputs.

For H5AD input, each tissue directory should contain:

```text
data/<tissue>/train.h5ad
data/<tissue>/test.h5ad
```

For Parquet input, each tissue directory should contain:

```text
data/<tissue>/train_parquets/
data/<tissue>/test_parquets/
data/<tissue>/train_ontology_to_int.json
data/<tissue>/train_celltype_to_int.json
data/<tissue>/all_ontology_to_int.json
data/<tissue>/all_celltype_to_int.json
```

### Run H5AD Workflow

```bash
python analyses/within_tissue_annotation/run_scLEAD.py \
  --data-dir data \
  --tissues left_lung \
  --save-root outputs/within_tissue_annotation
```

### Run Parquet Workflow

```bash
bash analyses/within_tissue_annotation/run_scLEAD.sh
```

To override the default paths, set the corresponding environment variables:

```bash
DATA_DIR=/path/to/parquet_tissues \
SAVE_ROOT=outputs/within_tissue_annotation \
bash analyses/within_tissue_annotation/run_scLEAD.sh
```

## Foundation-Model Training

### Input

The foundation-model training workflow expects one dataset directory containing:

```text
train_parquets/
train_ontology_to_int.json
```

### Run

```bash
python analyses/zero_shot_annotation/train_foundation_model.py \
  --data-dir data/<dataset_name> \
  --save-dir outputs/foundation_model \
  --graph-emb-npz data/graph_embeddings/cl_poincare_embeddings.npz
```

## Zero-Shot Prediction

### Input

This workflow requires:

```text
data/zeroshot-data/<dataset>.h5ad
```

A trained checkpoint must be provided explicitly using `--ckpt-path`.

### Run

```bash
python analyses/zero_shot_annotation/run_zero_shot_prediction.py \
  --ckpt-path outputs/foundation_model/best.ckpt \
  --data-dir data/zeroshot-data \
  --datasets GSE111976 \
  --output-dir outputs/zero_shot_annotation
```

## Cell Clustering

### Input

This workflow requires:

```text
data/<tissue>/test.h5ad
```

By default, the workflow uses the tissue list CSV at:

```text
data/tissue_cell_counts.csv
```

A different tissue list can be provided with `--tissue_csv`.

### Run

```bash
python analyses/cell_clustering/run_clustering_scLEAD_new.py \
  --ckpt_path outputs/foundation_model/best.ckpt \
  --data_dir data \
  --result_dir outputs/cell_clustering \
  --tissue_csv data/tissue_cell_counts.csv \
  --normalize_before_model \
  --normalize_embedding
```

## Notes

* `src/sclead/ct_descriptions/` contains packaged metadata and should remain in the source tree.
* Large raw datasets, checkpoints, and generated result folders should not be stored under `analyses/`.
* If datasets or outputs are stored outside the repository, provide absolute paths using `--data-dir`, `--save-dir`, `--output-dir`, or the appropriate environment variables instead of modifying the scripts.
