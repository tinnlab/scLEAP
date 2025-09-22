# scLMCT

**scLMCT: Single‑Cell Language‑Model‑Guided Cell‑Type Annotation**
abstract

> **TL;DR** Train scLMCT on donor‑split tissue data, evaluate against baselines (scGPT, scTab, TOSICA, …), and regenerate all tables/figures with one command.

---

## Features

* **Train a new cell type annotation model** 
* **zezo-shot cell annotation using pretrained model** 
* **reproducibility of all results in the manuscript** 

---

## Getting Started

### 1) Prerequisites

* Linux/macOS, Python ≥ 3.10
* (Recommended) Conda/Mamba for env management
* NVIDIA GPU + CUDA/cuDNN 

```bash
# (Optional) Create the exact research environment
mamba env create -f ./experiments/configs.yml  # or: conda env create -f environment.yml
conda activate scLMCT
```


### 2) Clone & Install sclmct

```bash
git clone https://github.com/tinnlab/scLMCT.git
cd scLMCT

pip install -e .
```

---

## Data

Place data under `./data/` using the layout below. Replace `DOWNLOAD_LINK_*` with your actual sources (e.g., Zenodo, Figshare, CELLxGENE Census, GEO mirrors).

```
scLMCT/
 ├─ data/
 │   ├─ cellxgene/              # cellxgene data
 │   ├─ zero_shot_data/         # external data for zero-shot annotation
 └─ ...
```

**Download:**

* cellxgene: **\[DOWNLOAD\_LINK\_TRAINING](https://seafile.tinnguyen-lab.com/d/91f5b323e38645d28808/)** → put under `data/cellxgene/`
* zero_shot_data: **\[DOWNLOAD\_LINK\_EXTERNAL](https://seafile.tinnguyen-lab.com/d/4b817456e18f4add8c48/)** → `data/zero_shot_data/`

---

## Quickstart

### Training a new cell type annotation model
The tutorial for training new cell type annotation model is given in [this notebook](https://github.com/tinnlab/scLMCT/tree/main/experiments/cell_type_annotation/scLMCT/training_new_annotation_model.ipynb)

### Zero-shot cell type annotation
The tutorial for running zero-shot annotation is given in [this notebook](https://github.com/tinnlab/scLMCT/tree/main/experiments/cell_type_annotation/scLMCT/zero_shot_annotation.ipynb)


---

## Reproducibility

* All experiments fix seeds and deterministic backends where feasible.
* Metrics are computed **per‑donor, per‑cell‑type** and saved as a dataframe.
* Figures are regenerated directly from result CSVs.

### A) Re‑run the full analysis

1. Ensure data are in `./data` (see **Data** section).
2. Run the desired method script, e.g.:

   ```bash
   bash ./scripts/run_scGPT.sh
   ```

### B) Reproduce all scores & figures from released artifacts

1. **(Optional)** Download precomputed results: **\[DOWNLOAD\_LINK\_RESULTS]** → put under `./results`
2. Collect metrics into a single CSV:

   ```bash
   python ./experiments/metrics_collect.py \
     --results_dir ./results \
     --out_csv ./results/summary_metrics.csv
   ```
3. Generate all paper figures:

   ```bash
   python ./experiments/plottings.py \
     --metrics_csv ./results/summary_metrics.csv \
     --out_dir ./figs
   ```

Expected long‑table schema (written by the collectors):

```
['method','tissue','donor_id','cell_type','metric','score']
```

---


## Repository Structure

```
scLMCT/
 ├─ configs/                    # YAML configs for experiments
 ├─ scripts/                    # Bash entrypoints per method
 ├─ src/                        # Main code of the package
 ├─ experiments/                # Code for comparison methods & metric collection & plotting
 ├─ results/                    # outputs (checkpoints, CSVs, figs)
 ├─ LICENSE                     # choose a license (see below)
 └─ README.md                   # this file
```

---


## Hardware & Runtime (guidance)

* **GPU**: ≥ 16 GB recommended for full‑scale runs (works on smaller with gradient accumulation)
* **CPU RAM**: ≥ 32 GB suggested for large AnnData operations
* **Disk**: 100–500 GB depending on cached datasets and artifacts

---

## Troubleshooting

* **`faiss` install issues**: prefer `conda install faiss-gpu -c pytorch` to match CUDA.
* **AnnData read errors**: ensure `anndata` ≥ 0.10 and `h5py` compiled with HDF5 ≥ 1.12.
* **Reproducibility drift**: confirm seeds and set `torch.backends.cudnn.deterministic=True`.
* **Plot fonts/fig sizes**: pass `--style paper` to `experiments/plottings.py`.

---

## Citation


## License


## Contributing


## Contact
