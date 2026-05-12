# Within-Tissue Annotation

This directory contains the main scLEAD training and evaluation workflow for within-tissue annotation experiments.

## Installation

Install the package in editable mode from the repository root:

```bash
pip install -e .
```

If you want the full GPU research environment, use `sclead.yml`.

## Entry Point

```bash
python analyses/within_tissue_annotation/run_scLEAD.py --help
```

The script supports both H5AD- and parquet-based datasets and uses the packaged ontology metadata in `src/sclead/ct_descriptions/`.
