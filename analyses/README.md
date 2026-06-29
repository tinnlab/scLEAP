# Analyses Guide

This directory contains runnable analysis workflows. The scripts now resolve repo-relative paths, so you can run them from the repository root without depending on one specific working directory.

## Data Layout

Put local datasets under [data](scLEAD/data):

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

Small shared metadata already used by the analyses lives in [analyses/data](https://github.com/tinnlab/scLEAD/blob/main/analyses/data):

- [analyses/data/tissue_cell_counts.csv](https://github.com/tinnlab/scLEAD/blob/main/data/tissue_cell_counts.csv)
- [analyses/data/graph_embeddings/cl_poincare_embeddings.npz](https://github.com/tinnlab/scLEAD/blob/main/data/graph_embeddings/cl_poincare_embeddings.npz)

Write generated outputs under `outputs/` instead of back into `analyses/`.

## Within-Tissue Annotation

Input:
- H5AD mode: `data/<tissue>/train.h5ad` and `data/<tissue>/test.h5ad`
- Parquet mode: `data/<tissue>/train_parquets/`, `test_parquets/`, and the mapping JSON files

Run:

```bash
python analyses/within_tissue_annotation/run_scLEAD.py \
  --data-dir data \
  --tissues left_lung \
  --save-root outputs/within_tissue_annotation
```

Parquet runner:

```bash
bash analyses/within_tissue_annotation/run_scLEAD.sh
```

Override defaults when needed:

```bash
DATA_DIR=/path/to/parquet_tissues \
SAVE_ROOT=outputs/within_tissue_annotation \
bash analyses/within_tissue_annotation/run_scLEAD.sh
```

## Foundation-Model Training

Input:
- One dataset folder containing `train_parquets/` and `train_ontology_to_int.json`

Run:

```bash
python analyses/zero_shot_annotation/train_foundation_model.py \
  --data-dir data/<dataset_name> \
  --save-dir outputs/foundation_model \
  --graph-emb-npz analyses/data/graph_embeddings/cl_poincare_embeddings.npz
```

## Zero-Shot Prediction

Input:
- `data/zeroshot-data/<dataset>.h5ad`
- A checkpoint path supplied explicitly

Run:

```bash
python analyses/zero_shot_annotation/run_zero_shot_prediction.py \
  --ckpt-path outputs/foundation_model/best.ckpt \
  --data-dir data/zeroshot-data \
  --datasets GSE111976 \
  --output-dir outputs/zero_shot_annotation
```

## Cell Clustering

Input:
- `data/<tissue>/test.h5ad`
- Tissue list CSV at [analyses/data/tissue_cell_counts.csv](https://github.com/tinnlab/scLEAD/blob/main/data/tissue_cell_counts.csv), or pass a different one

Run:

```bash
python analyses/cell_clustering/run_clustering_scLEAD_new.py \
  --ckpt_path outputs/foundation_model/best.ckpt \
  --data_dir data \
  --result_dir outputs/cell_clustering \
  --tissue_csv analyses/data/tissue_cell_counts.csv \
  --normalize_before_model \
  --normalize_embedding
```

## Notes

- `src/sclead/ct_descriptions/` contains packaged metadata and should stay in the source tree.
- Large raw data, checkpoints, and generated result folders should stay out of `analyses/`.
- If you keep external datasets elsewhere, pass absolute paths with `--data-dir`, `--save-dir`, `--output-dir`, or environment variables instead of editing the scripts.
