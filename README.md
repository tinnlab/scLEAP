# scLMCT

**This is the code base for scLMCT: Single‑Cell Language‑Model‑Guided Cell‑Type Annotation**


---

## Features

* **Train a cell type annotation model** 
* **Zero-shot cell annotation using pretrained model** 
* **Reproducibility of all results in the manuscript** 

---

## Getting Started

### 1) Prerequisites

* Linux/macOS, Python ≥ 3.10
* (Recommended) Conda/Mamba for env management
* NVIDIA GPU + CUDA/cuDNN 

```bash
# (Optional) Create the exact research environment
mamba env create -f ./experiments/configs/scLMCT.yml  
conda activate scLMCT
```


### 2) Clone & Install sclmct

```bash
git clone https://github.com/tinnlab/scLMCT.git
cd scLMCT

pip install -e .
```

---

## Quickstart

### Training a new cell type annotation model
The tutorial for training new cell type annotation model is given in [this notebook](https://github.com/tinnlab/scLMCT/tree/main/experiments/cell_type_annotation/scLMCT/training_new_annotation_model.ipynb)

### Zero-shot cell type annotation
The tutorial for running zero-shot annotation is given in [this notebook](https://github.com/tinnlab/scLMCT/tree/main/experiments/cell_type_annotation/scLMCT/zero_shot_annotation.ipynb)


---

## Data & Pre-trained weights

Place data under `./data/` using the layout below.

```
scLMCT/
 ├─ data/
 │   ├─ cellxgene/              # cellxgene data
 │   ├─ zero_shot_data/         # external data for zero-shot annotation
 └─ ...
```

**Download:**
* cellxgene: **[DOWNLOAD\_LINK\_CELLxGENE](https://seafile.tinnguyen-lab.com/d/91f5b323e38645d28808/)** → put under `data/cellxgene/`
* zero_shot_data: **[DOWNLOAD\_LINK\_ZERO\_SHOT](https://seafile.tinnguyen-lab.com/d/4b817456e18f4add8c48/)** → `data/zero_shot_data/`

Detail information of the datasets used in the manuscript can be found [HERE](https://seafile.tinnguyen-lab.com/f/512e913575a24058816a/) 

Everything else can be found [HERE](https://seafile.tinnguyen-lab.com/d/c86371665e324df7be53/)

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

1. **(Optional)** All analyses results can be found **[HERE](https://seafile.tinnguyen-lab.com/d/2d3acb0834534795a473/)** → put under `./results`
2. Collect metrics into a single CSV:

   ```bash
   python ./experiments/metrics_collect.py 
   ```

Expected two result tables given **[HERE](https://seafile.tinnguyen-lab.com/d/2d3acb0834534795a473/)**

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
All analyses were run in linux environment with the following specification:
* **GPU**: single A6000 (48GB)
* **CPU RAM**: ≥ 32 GB suggested for large AnnData operations
* **Disk**: 100–500 GB depending on cached datasets and artifacts

---


---

## Citation


## License


## Contributing


## Contact
