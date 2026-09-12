# Within-Tissue Annotation

This directory contains the main scLEAP training and evaluation workflow for within-tissue annotation experiments.

## Installation

Install the package in editable mode from the repository root:

```bash
pip install -e .
```

If you want the full GPU research environment, use `scleap.yml`.

## Entry Point

```bash
python analyses/within_tissue_annotation/run_scLEAP.py --help
```

The script supports both H5AD- and parquet-based datasets and uses the packaged ontology metadata in `src/scleap/ct_descriptions/`.
