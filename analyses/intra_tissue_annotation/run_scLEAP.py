#%%
#%%
"""
Ablation / evaluation pipeline for scLEAP (CLIP-style single-cell model)
------------------------------------------------------------------------
- Robust CLI with sensible defaults
- Clear separation of concerns
- Safer mapping logic (ontology vs. celltype)
- GPU/CPU FAISS fallback with **native GPU index**, FP16, chunked add/search
- Deterministic seeding and threading control
- Proactive CUDA memory release before FAISS build/query
- Works with Parquet or H5AD backends

Assumes the following local modules exist:
  - model.TrainWrapperCLIPStyle
  - dataset.H5ADDataset, LOG1PTransform, TotalSumNormalize
  - utils.predict_with_distance_weights
  - gpu_parquet_window_loader.build_loader

Author: cleaned & refactored + FAISS/GPU mem improvements
"""
from __future__ import annotations

import argparse
import ast
import gc
import json
import multiprocessing as mp
import os
import time
import warnings
from dataclasses import dataclass
from os.path import join
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import scanpy as sc
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, classification_report, f1_score

# External deps
from faiss import IndexFlatIP, normalize_L2  # type: ignore
import faiss  # type: ignore

# Project-local deps (must be available on PYTHONPATH)
# from model_v1 import TrainWrapperCLIPStyle
from scleap.model import TrainWrapperCLIPStyle

from scleap.dataset import H5ADDataset, LOG1PTransform, TotalSumNormalize
from scleap.dataset_gpu import H5ADDatasetGPU
from scleap.gpu_parquet_dataset import build_loader

from scleap.utils import predict_with_distance_weights, load_cell_types_info, load_cell_types_mapping

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_GRAPH_EMB_NPZ = str(REPO_ROOT / "data" / "graph_embeddings" / "cl_poincare_embeddings.npz")
DEFAULT_OLS_MAPPINGS = str(REPO_ROOT / "src" / "scleap" / "ct_descriptions" / "cell_name_to_ols_id.json")
DEFAULT_CT_DESCRIPTIONS = str(REPO_ROOT / "src" / "scleap" / "ct_descriptions" / "cell_types_info.json")
# ----------------------------
# Utilities & Config
# ----------------------------

THREAD_ENV_VARS = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "BLAS_NUM_THREADS": "1",
}


@dataclass
class Args:
    # IO & data
    data_dir: str
    save_root: str = "./results"
    ols_mappings_file: str = DEFAULT_OLS_MAPPINGS
    ct_description_file: str = DEFAULT_CT_DESCRIPTIONS
    use_parquet: bool = True
    external_data: bool = False

    # Model / training
    text_encoder_model: str = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"
    n_cell_layers: int = 5
    n_text_layers_to_finetune: int = 1
    n_prompts: int = 5
    pool_type: str = "mean"
    hidden_dim: int = 768
    use_projection_head: bool = False

    # Loss / logits hyperparams
    s: float = 30.0
    m1: float = 1.0
    m2: float = 0.3
    m3: float = 0.1
    m4: float = 1.5
    f_gamma: float = 1.3
    l_ct: float = 1.0
    l_tt: float = 0.1
    l_contrastive: float = 0.1
    l_graph: float = 0.1
    graph_emb_npz: str = DEFAULT_GRAPH_EMB_NPZ


    # Training setup
    seed: int = 1
    max_epoch: int = 200
    gradient_clip_val: float = 1.0

    # Execution
    parallel: bool = True
    workers: int = 8
    gpu_id: int = 0  # index into CUDA_VISIBLE_DEVICES
    cuda_visible: str | None = None  # e.g. "1,2"; if None, do not override

    # Eval / aux
    classifier: str = "faiss"
    fold: int = 0
    test: bool = False
    test_tissue: str = "temporal cortex"
    tissues: List[str] | None = None


# ----------------------------
# System prep
# ----------------------------

def setup_environment_threads() -> None:
    for k, v in THREAD_ENV_VARS.items():
        os.environ[k] = v


def set_cuda_visible(devs: str | None) -> None:
    if devs is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = devs


def resolve_repo_path(path_str: str) -> str:
    path = Path(path_str).expanduser()
    if path.is_absolute():
        return str(path)
    return str((REPO_ROOT / path).resolve())


# ----------------------------
# Data helpers
# ----------------------------

def determine_batch_size(n: int) -> int:
    if n > 100_000:
        return 4096
    if n > 50_000:
        return 2048
    if n > 10_000:
        return 2048
    if n > 1_000:
        return 128
    if n < 64:
        return 16
    return 64


def determine_batch_size_parquet(n: int) -> int:
    if n == 1:
        return 128
    if n <=4:
        return 2048
    else:
        return 4096 * 2



def get_all_parquets(tissue_dir: str) -> Tuple[List[str], List[str]]:
    """Return lists of parquet file paths for train/test under a tissue dir."""
    train_glob = join(tissue_dir, "train_parquets")
    test_glob = join(tissue_dir, "test_parquets")

    train_parquets, test_parquets = [], []
    if os.path.isdir(train_glob):
        for f in os.listdir(train_glob):
            if f.endswith(".parquet"):
                train_parquets.append(join(train_glob, f))
    if os.path.isdir(test_glob):
        for f in os.listdir(test_glob):
            if f.endswith(".parquet"):
                test_parquets.append(join(test_glob, f))
    return train_parquets, test_parquets


def load_mappings_for_parquet(data_dir: str, tissue: str):
    """Load ontology/celltype mapping dicts for a parquet-based tissue."""
    base = f"{data_dir}/{tissue}"
    with open(f"{base}/train_ontology_to_int.json") as f:
        train_ontology_to_int = json.load(f)
    with open(f"{base}/train_celltype_to_int.json") as f:
        train_celltype_to_int = json.load(f)
    with open(f"{base}/all_ontology_to_int.json") as f:
        all_ontology_to_int = json.load(f)
    with open(f"{base}/all_celltype_to_int.json") as f:
        all_celltype_to_int = json.load(f)

    # Normalize cases/underscores for safety
    train_ontology_to_int = {k.lower(): v for k, v in train_ontology_to_int.items()}
    train_celltype_to_int = {k.lower().replace("_", " "): v for k, v in train_celltype_to_int.items()}
    all_ontology_to_int = {k.lower().replace("_", " "): v for k, v in all_ontology_to_int.items()}
    all_celltype_to_int = {k.lower(): v for k, v in all_celltype_to_int.items()}

    train_int_to_ontology = {v: k for k, v in train_ontology_to_int.items()}
    train_int_to_celltype = {v: k for k, v in train_celltype_to_int.items()}
    all_int_to_ontology = {v: k for k, v in all_ontology_to_int.items()}
    all_int_to_celltype = {v: k for k, v in all_celltype_to_int.items()}

    return (
        train_ontology_to_int,
        train_celltype_to_int,
        all_ontology_to_int,
        all_celltype_to_int,
        train_int_to_ontology,
        train_int_to_celltype,
        all_int_to_ontology,
        all_int_to_celltype,
    )


# ----------------------------
# Text prompts for label semantics
# ----------------------------

def get_semantics_labels(
    ct_description_file: str,
    train_label_dict: Dict[str, int],
    mappings: Dict[str, str] | None = None,
) -> Dict[int, List[str]]:
    """Build a dict: class_id -> list[prompt strings].

    - If a curated description file is provided, use it; otherwise fall back to
      simple templates. When `mappings` is given, it maps keys in the json file
      to the label space of `train_label_dict`.
    """
    simple_templates = [
        "A single-cell transcriptome from a {label} cell.",
        "This is the gene expression profile of a {label} cell.",
        "Cell type: {label} cell.",
        "Based on its gene expression, this cell is a {label} cell.",
        "A biologically annotated cell profile: {label} cell.",
        "An scRNA-seq profile labeled as: {label} cell.",
        "This cell's identity is: {label} cell",
        "This cell is classified as a {label} cell.",
        "Transcriptomic identity: {label} cell.",
        "This vector encodes the functional signature of a {label} cell.",
    ]

    try:
        # with open(ct_description_file, "r") as f:
        #     label_prompts = json.load(f)
        label_prompts = load_cell_types_info()

        if mappings is not None:
            filtered = {k: v for k, v in label_prompts.items() if k in mappings and mappings[k] in train_label_dict}
            label_prompts = {train_label_dict[mappings[k]]: v for k, v in filtered.items()}
        else:
            label_prompts = {
                train_label_dict[k.lower()]: v
                for k, v in label_prompts.items()
                if k.lower() in train_label_dict
            }

        semantics_labels = {k: label_prompts[k] for k in sorted(label_prompts.keys())}

        # Add unknown prompts if relevant
        if "unknown" in train_label_dict:
            semantics_labels[train_label_dict["unknown"]] = [
                "This unknown cell population does not match any known cell type and may represent a novel or intermediate state.",
                "This unknown cell population expresses mixed lineage markers and may represent a novel or intermediate state.",
                "This unknown cell population has ambiguous classification with low confidence, suggesting a novel or intermediate state.",
                "This unknown cell population is uncharacterized and may represent a rare or intermediate state.",
                "This unknown cell population lacks canonical markers and may represent a novel or intermediate state.",
                "This unknown cell population is not captured in current ontologies and may represent a novel or intermediate state.",
                "This unknown cell population partially resembles immune subtypes but remains unclassified, suggesting a novel or intermediate state.",
                "This unknown cell population forms a distinct cluster and may represent a novel or intermediate state.",
                "This unknown cell population lacks unique markers and may represent a heterogeneous or intermediate state.",
                "This unknown cell population may reflect stress, doublets, or noise, and is best described as a novel or intermediate state.",
            ]

    except Exception:
        # Fallback: synthesize prompts from labels
        semantics_labels = {}
        for label, class_id in train_label_dict.items():
            base = label.replace("cell", "").strip()
            semantics_labels[class_id] = [t.format(label=base) for t in simple_templates]

    return semantics_labels


# ----------------------------
# Embeddings & Metrics
# ----------------------------

def to_ols_id(label: str, mappings: Dict[str, str]) -> str:
    """Return an OLS id for a label.
    - If label already looks like an OLS id (e.g., 'cl:0000057'), pass through.
    - Else try to map via provided name->OLS dict; fallback to 'unknown'.
    """
    lab = str(label).strip().lower()
    if lab.startswith("cl:"):
        return lab
    return mappings.get(lab, "unknown")

def get_latent_representations(
    data_loader,
    encoder: torch.nn.Module,
    label_mapping: Dict[int, str],
    device: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """Run encoder on a dataloader and collect (Z, labels)."""
    encoder = encoder.to(device)
    encoder.eval()

    latents: List[np.ndarray] = []
    labels: List[str] = []

    with torch.no_grad():
        for x, y in data_loader:
            if isinstance(x, dict):
                x = x["X"]
            x = x.to(device)
            z = encoder(x)
            latents.append(z.detach().cpu().numpy())
            labels.extend([label_mapping[int(i)].replace("_", " ").lower() for i in y.detach().cpu().numpy()])

    return np.concatenate(latents, axis=0), np.array(labels)


def compute_metrics(y_true: Sequence, y_pred: Sequence) -> Dict[str, float]:
    print(classification_report(y_true, y_pred, zero_division=0))
    return {
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "F1 Weighted": float(f1_score(y_true, y_pred, average="weighted")),
        "F1 Macro": float(f1_score(y_true, y_pred, average="macro")),
    }


# ----------------------------
# FAISS helpers (fast + memory friendly)
# ----------------------------

def maybe_close_loader(loader) -> None:
    """Best-effort: close dataset/loader if it exposes a shutdown/close API."""
    if loader is None:
        return
    # Try dataset-level close first
    try:
        ds = getattr(loader, "dataset", None)
        for attr in ("close", "shutdown", "stop"):
            if ds is not None and hasattr(ds, attr):
                try:
                    getattr(ds, attr)()
                except Exception:
                    pass
    except Exception:
        pass
    # Then try loader itself
    for attr in ("close", "shutdown", "stop"):
        if hasattr(loader, attr):
            try:
                getattr(loader, attr)()
            except Exception:
                pass


def release_cuda_memory(*objs) -> None:
    """Aggressively free CUDA memory held by Python refs."""
    for o in objs:
        try:
            del o
        except Exception:
            pass
    try:
        torch.cuda.synchronize()
    except Exception:
        pass
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass
    gc.collect()


def create_faiss_index_fast(
    vectors: np.ndarray,
    use_gpu: bool = True,
    gpu_id: int = 0,
    use_fp16: bool = True,
    add_batch: int = 262_144,
):
    """
    Create a cosine-sim (IP on L2-normalized) index.
    - On GPU: native GpuIndexFlatIP with optional FP16 and chunked add().
    - On CPU: IndexFlatIP with chunked add().
    """
    d = vectors.shape[1]
    if use_gpu:
        try:
            res = faiss.StandardGpuResources()
            cfg = faiss.GpuIndexFlatConfig()
            cfg.device = gpu_id
            cfg.useFloat16 = use_fp16
            index = faiss.GpuIndexFlatIP(res, d, cfg)
        except Exception as e:
            warnings.warn(f"GPU FAISS unavailable ({e}); falling back to CPU.")
            use_gpu = False
    if not use_gpu:
        index = IndexFlatIP(d)

    vecs = vectors.astype(np.float32, copy=False)
    n = vecs.shape[0]
    for start in range(0, n, add_batch):
        stop = min(start + add_batch, n)
        index.add(vecs[start:stop])
    return index


def faiss_search_batched(index, queries: np.ndarray, k: int = 1, batch: int = 262_144):
    """Chunked search to bound temp mem on CPU/GPU."""
    Q = queries.astype(np.float32, copy=False)
    n = Q.shape[0]
    Ds, Is = [], []
    for start in range(0, n, batch):
        stop = min(start + batch, n)
        D, I = index.search(Q[start:stop], k)
        Ds.append(D); Is.append(I)
    return np.vstack(Ds), np.vstack(Is)


# ----------------------------
# Core per-tissue run
# ----------------------------

def get_path_to_save(args: Args, tissue: str) -> Tuple[str, str]:
    base_path = f"{args.save_root}"
    if args.external_data:
        save_path = os.path.join(base_path, tissue, f"fold_{args.fold}")
    else:
        save_path = os.path.join(base_path, tissue)
    return save_path, base_path


def run_tissue(tissue: str, args: Args) -> None:
    start_t0 = time.time()
    print(f"\n=== Running tissue: {tissue} ===")

    # Determinism
    torch.set_float32_matmul_precision("high")
    torch.use_deterministic_algorithms(True, warn_only=True)

    # Keep FAISS OMP threads in check (avoids oversubscription)
    try:
        faiss.omp_set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "1")))
    except Exception:
        pass

    path_to_save, _ = get_path_to_save(args, tissue)
    os.makedirs(path_to_save, exist_ok=True)

    # Load ontology term mapping
    # with open(args.ols_mappings_file, "r") as f:
    #     ols_mappings = json.load(f)
    ols_mappings = load_cell_types_mapping()
    ols_mappings = {k.lower(): v.lower() for k, v in ols_mappings.items()}
    inverse_ols_mappings = {v: k for k, v in ols_mappings.items()}


    ## read the graph embeddings

    ## dataframe with columns "name" "cell_label" "embedding"
    # data = np.load("./extracted_embeddings/graph_node_embs.npz", allow_pickle=True)
    data = np.load(args.graph_emb_npz, allow_pickle=True)
    names = data["names"]            # shape: [N]
    embeddings = data["embeddings"]  # shape: [N, D], float32
    # optional: build a lookup dict
    name_to_idx = {name.lower(): i for i, name in enumerate(names)}

    # --------------------
    # Build loaders & label dicts
    # --------------------
    start_data = time.time()

    if args.use_parquet:
        train_parquets, test_parquets = get_all_parquets(f"{args.data_dir}/{tissue}")
        (
            train_ontology_to_int,
            train_celltype_to_int,
            all_ontology_to_int,
            all_celltype_to_int,
            train_int_to_ontology,
            train_int_to_celltype,
            all_int_to_ontology,
            all_int_to_celltype,
        ) = load_mappings_for_parquet(args.data_dir, tissue)

        # Labels in parquet are *ontology* ints
        columns = ["X", "ontology_int"]
        list_columns = ["X"]
        label_columns = ["ontology_int"]
        casts = {"ontology_int": "int64"}

        # Batch size: use larger when many parts
        batch_size = determine_batch_size_parquet(len(train_parquets))
        window_rows = 8192*2
        rmm_pool = "off"

        train_loader = build_loader(
            files=train_parquets,
            columns=columns,
            list_columns=list_columns,
            label_columns=label_columns,
            casts=casts,
            batch_size=batch_size,
            window_rows=window_rows,
            shuffle=True,
            part_shuffle=True,
            drop_last=True,
            seed=args.seed,
            device_id=args.gpu_id,
            rmm_pool=rmm_pool,
        )

        train_loader_eval = build_loader(
            files=train_parquets,
            columns=columns,
            list_columns=list_columns,
            label_columns=label_columns,
            casts=casts,
            batch_size=batch_size,
            window_rows=window_rows,
            shuffle=False,
            part_shuffle=False,
            drop_last=False,
            seed=args.seed,
            device_id=args.gpu_id,
            rmm_pool=rmm_pool,
        )

        test_loader = build_loader(
            files=test_parquets,
            columns=columns,
            list_columns=list_columns,
            label_columns=label_columns,
            casts=casts,
            batch_size=batch_size,
            window_rows=window_rows,
            shuffle=False,
            part_shuffle=False,
            drop_last=False,
            seed=args.seed,
            device_id=args.gpu_id,
            rmm_pool=rmm_pool,
        )

        # Build prompt set over the *ontology* label space
        semantics_labels = get_semantics_labels(
            args.ct_description_file,
            train_ontology_to_int,
            mappings=None,
        )
        # print(semantics_labels)

        with open(f"{args.data_dir}/{tissue}/train_ontology_to_int.json", 'r') as f:
            train_label_dict = json.load(f)
        train_label_dict = {k.lower(): v for k, v in train_label_dict.items()}
        print(train_label_dict)

         ## get graph embeddings for only the training cells
        graph_embeddings = np.zeros((len(train_label_dict), embeddings.shape[1]), dtype=np.float32)
        for cell_ols_id, int_idx in train_label_dict.items():
            if cell_ols_id in name_to_idx:
                graph_embeddings[int_idx] = embeddings[name_to_idx[cell_ols_id]]

        print("Embeddings for training cells shape:", graph_embeddings.shape)


        # Label id -> string mapping for dataloaders and semantic class ids
        label_id_to_name_for_train = train_int_to_ontology
        label_id_to_name_for_train_all = all_int_to_ontology
        label_id_to_name_for_test = all_int_to_ontology
        label_id_to_name_for_sem = all_int_to_ontology

    else:
        path_train = f"{args.data_dir}/{tissue}/train.h5ad"
        path_test = f"{args.data_dir}/{tissue}/test.h5ad"

        transform = torch.nn.Sequential(LOG1PTransform(), TotalSumNormalize(target_sum=1e4))

        # train_ds = H5ADDataset(path_train, use_obs_column="cell_type", transform=transform, ols_mappings=ols_mappings)
        # train_label_dict = train_ds.label_dict  # str -> int
        train_ds = H5ADDatasetGPU(path_train, use_obs_column="cell_type", ols_mappings=ols_mappings, transform=transform, device="cuda")
        train_label_dict = train_ds.label_dict

        # Prompts over training label space
        semantics_labels = get_semantics_labels(args.ct_description_file, train_label_dict, mappings=None)
        # print(semantics_labels)
        print(train_label_dict)

        ## get graph embeddings for only the training cells
        graph_embeddings = np.zeros((len(train_label_dict), embeddings.shape[1]), dtype=np.float32)
        for cell_ols_id, int_idx in train_label_dict.items():
            if cell_ols_id in name_to_idx:
                graph_embeddings[int_idx] = embeddings[name_to_idx[cell_ols_id]]

        print("Embeddings for training cells shape:", graph_embeddings.shape)


        batch_size = determine_batch_size(len(train_ds))
        g = torch.Generator().manual_seed(args.seed)

        from torch.utils.data import DataLoader

        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
            worker_init_fn=lambda i: np.random.seed(args.seed + i),
            generator=g,
        )

        train_ds_eval = H5ADDataset(path_train, use_obs_column="cell_type", transform=transform, ols_mappings=ols_mappings)

        train_loader_eval = DataLoader(
            train_ds_eval,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            worker_init_fn=lambda i: np.random.seed(args.seed + i),
            generator=g,
        )
        test_ds = H5ADDataset(path_test, use_obs_column="cell_type", transform=transform, ols_mappings=ols_mappings)

        test_loader = DataLoader(
            test_ds,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            worker_init_fn=lambda i: np.random.seed(args.seed + i),
            generator=g,
        )

        # Index->label mappers (ontology names)
        label_id_to_name_for_train = train_ds.get_index_to_label_mapping()
        label_id_to_name_for_test = test_ds.get_index_to_label_mapping()
        label_id_to_name_for_sem = train_ds.get_index_to_label_mapping()

    # Log data processing time
    with open(join(path_to_save, "time_log.txt"), "a") as f:
        f.write(f"Data processing time: {time.time() - start_data:.2f} sec\n")

    # --------------------
    # Create / load model
    # --------------------
    start_model = time.time()

    # Peek feature dimension from one batch
    first_batch = next(iter(train_loader))
    if isinstance(first_batch[0], dict):
        d_input = first_batch[0]["X"].shape[-1]
    else:
        d_input = first_batch[0].shape[-1]

    ckpt_dir = join(path_to_save, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    cache_path = join(path_to_save, "cached_prompt_embeddings.pt")

    checkpoint_paths = [join(ckpt_dir, f) for f in os.listdir(ckpt_dir) if f.endswith(".ckpt")]
    if len(checkpoint_paths) < 0: ## currently disabled
        # Load the first / best checkpoint
        checkpoint_path = sorted(checkpoint_paths)[0]
        print(f"Found checkpoint. Loading from {checkpoint_path}")
        model = TrainWrapperCLIPStyle.load_from_checkpoint(checkpoint_path)
    else:
        print("No checkpoint found. Training a new model…")
        model = TrainWrapperCLIPStyle(
            input_dim=d_input,
            hidden_dim=args.hidden_dim,
            output_dim=None,
            label_prompts=semantics_labels,
            n_cell_layers=args.n_cell_layers,
            text_encoder_model=args.text_encoder_model,
            use_projection_head=args.use_projection_head,
            n_prompts=args.n_prompts,
            pool_type=args.pool_type,
            n_text_layers_to_finetune=args.n_text_layers_to_finetune,
            device="cuda",
            cache_path=cache_path,
            lr=1e-4,
            weight_decay=1e-4,
            s=args.s,
            m1=args.m1,
            m2=args.m2,
            m3=args.m3,
            m4=args.m4,
            f_gamma=args.f_gamma,
            l_ct=args.l_ct,
            l_tt=args.l_tt,
            l_contrastive=args.l_contrastive,
            l_graph=args.l_graph,
            graph_embeddings=graph_embeddings
        )

        callbacks = [
            pl.callbacks.EarlyStopping(monitor="train_loss", patience=5, min_delta=1e-3),
            pl.callbacks.ModelCheckpoint(dirpath=ckpt_dir, monitor="train_loss", save_top_k=1, mode="min", every_n_epochs=1),
        ]

        trainer = pl.Trainer(
            max_epochs=args.max_epoch,
            accelerator="gpu",
            devices=1,  # use currently visible GPU
            callbacks=callbacks,
            gradient_clip_val=args.gradient_clip_val,
            deterministic=True,
            enable_progress_bar=True,
        )
        trainer.fit(model, train_loader)

        # Reload best model
        best_path = callbacks[1].best_model_path if hasattr(callbacks[1], "best_model_path") else None
        if best_path and os.path.isfile(best_path):
            model = TrainWrapperCLIPStyle.load_from_checkpoint(best_path)
        print(f"Training finished in {time.time() - start_model:.2f} seconds.")

    with open(join(path_to_save, "time_log.txt"), "a") as f:
        f.write(f"Model creating and training time: {time.time() - start_model:.2f} sec\n")

    # --------------------
    # Latent inference
    # --------------------
    print("Inferring latent representations…")
    t_lat0 = time.time()

    # Training set latents + labels
    eval_loader_for_lat = train_loader_eval
   

    train_lat, train_labels_str = get_latent_representations(
        eval_loader_for_lat,
        model,
        label_mapping=label_id_to_name_for_train,
        device="cuda",
    )
    normalize_L2(train_lat)

    # Test set latents + labels
    test_lat, test_labels_str = get_latent_representations(
        test_loader,
        model,
        label_mapping=label_id_to_name_for_test,
        device="cuda",
    )
    normalize_L2(test_lat)

    print(f"Latents ready in {time.time() - t_lat0:.2f} seconds.")

    with open(join(path_to_save, "time_log.txt"), "a") as f:
        f.write(f"Latent inference time: {time.time() - t_lat0:.2f} sec\n")

    # --------------------
    # Text (class) embeddings
    # --------------------
    print("Encoding semantic (text) class centers…")
    t_sem0 = time.time()
    model.setup()
    with torch.no_grad():
        sem_lbls_tensor, sem_embeds = model.get_class_text_centers(only_centers=True)

    sem_lbls = sem_lbls_tensor.detach().cpu().numpy().tolist()  # ints over training label space
    sem_embeds = sem_embeds.detach().cpu().numpy()

    # Normalize for cosine similarity via inner product
    normalize_L2(sem_embeds)

    # Map class ids -> ontology strings
    sem_label_names = [label_id_to_name_for_sem[int(i)] for i in sem_lbls]
    sem_label_names = [s.lower().replace("_", " ") for s in sem_label_names]

    # Map training labels -> OLS ids (with fallback)
    train_labels_ols = np.array([to_ols_id(x, ols_mappings) for x in train_labels_str])

    with open(join(path_to_save, "time_log.txt"), "a") as f:
        f.write(f"Text embedding time: {time.time() - t_sem0:.2f} sec\n")

    # --------------------
    # FREE CUDA before FAISS build/search (critical for speed)
    # --------------------
    try:
        maybe_close_loader(train_loader)
        maybe_close_loader(train_loader_eval)
        maybe_close_loader(test_loader)
    except Exception:
        pass

    try:
        model.to("cpu")
    except Exception:
        pass

    objs_to_release = []
    for name in ("trainer", "first_batch", "train_loader", "train_loader_eval", "test_loader"):
        if name in locals():
            objs_to_release.append(locals()[name])
    release_cuda_memory(*objs_to_release)

    # --------------------
    # Build FAISS indices (cosine via IP on L2-normalized vectors)
    # --------------------
    print("Building FAISS indices…")
    t_faiss0 = time.time()

    # Normalize latents as well (cosine similarity)
    train_lat_norm = train_lat.astype(np.float32, copy=True)
    test_lat_norm = test_lat.astype(np.float32, copy=True)
    normalize_L2(train_lat_norm)
    normalize_L2(test_lat_norm)

    # Latent bank: big -> GPU native index with FP16 + chunked add (fallback to CPU if needed)
    try:
        latent_index = create_faiss_index_fast(
            train_lat_norm,
            use_gpu=True,
            gpu_id=args.gpu_id,
            use_fp16=True,
            add_batch=262_144,
        )
    except Exception as e:
        warnings.warn(f"GPU latent index failed ({e}); using CPU IndexFlatIP.")
        latent_index = create_faiss_index_fast(
            train_lat_norm, use_gpu=False, add_batch=262_144
        )

    # Semantic/text centers are small: keep on CPU to avoid extra GPU pressure
    semantic_index = IndexFlatIP(sem_embeds.shape[1])
    semantic_index.add(sem_embeds.astype(np.float32, copy=False))

    with open(join(path_to_save, "time_log.txt"), "a") as f:
        f.write(f"FAISS build time: {time.time() - t_faiss0:.2f} sec\n")

    # --------------------
    # Predict
    # --------------------
    print("Predicting labels with FAISS…")
    t_pred0 = time.time()

    # 1-NN over semantic/text centers (batched search to bound memory)
    D_sem, I_sem = faiss_search_batched(semantic_index, test_lat_norm, k=1, batch=262_144)
    pred_sem_names = [sem_label_names[i[0]] for i in I_sem]

    # k-NN over latent bank with distance weights (uses provided utility)
    pred_lat_ols, D_lat = predict_with_distance_weights(
        latent_index, test_lat_norm, train_labels_ols, k=11, return_distances=True
    )

    with open(join(path_to_save, "time_log.txt"), "a") as f:
        f.write(f"Predict time: {time.time() - t_pred0:.2f} sec\n")

    # --------------------
    # Build AnnData & metrics
    # --------------------
    print("Creating AnnData and computing metrics…")
    t_adata0 = time.time()

    # True labels -> OLS ids (vectorized)
    true_labels_ols = np.array([to_ols_id(x, ols_mappings) for x in test_labels_str])

    obs_df = pd.DataFrame(
        {
            "true_labels": test_labels_str,
            "true_labels_id": true_labels_ols,
            "pred_labels_faiss_sem_id": pred_sem_names,  # note: semantic predicts names, map to OLS below
            "pred_labels_faiss_lat_id": pred_lat_ols,
        }
    )

    # Map IDs <-> human-readable names using inverse mapping
    obs_df["pred_labels_faiss_sem"] = obs_df["pred_labels_faiss_sem_id"].map(lambda i: inverse_ols_mappings.get(i, i))
    obs_df["pred_labels_faiss_lat"] = obs_df["pred_labels_faiss_lat_id"].map(lambda i: inverse_ols_mappings.get(i, i))

    # Consider only classes seen in training if not external
    if args.external_data:
        mask_eval = np.ones(len(obs_df), dtype=bool)
    else:
        train_id_set = set(train_labels_ols)
        mask_eval = np.array([lbl in train_id_set for lbl in true_labels_ols])
    obs_df["idx2cal"] = False
    obs_df.loc[mask_eval, "idx2cal"] = True

    ## write the obs_df to csv
    obs_df.to_csv(join(path_to_save, "predictions.csv"), index=True)
    # pred_adata = sc.AnnData(X=test_lat_norm, obs=obs_df)
    # pred_adata.write_h5ad(join(path_to_save, "pred.h5ad"))

    # ## plot the umap of the embedding space
    # sc.pp.neighbors(pred_adata, use_rep='X', n_neighbors=15)
    # sc.tl.umap(pred_adata)
    # sc.settings.figdir = path_to_save
    # sc.pl.umap(pred_adata, color=['true_labels_id', 'pred_labels_faiss_lat_id'], save='umap_labels.png', show=False)

    # Metrics: use OLS ids for comparability
    y_true = obs_df.loc[mask_eval, "true_labels_id"].to_numpy()
    y_sem = obs_df.loc[mask_eval, "pred_labels_faiss_sem_id"].to_numpy()
    y_lat = obs_df.loc[mask_eval, "pred_labels_faiss_lat_id"].to_numpy()

    # print(y_true[:5], y_sem[:5], y_lat[:5])
    metrics_sem = compute_metrics(y_true, y_sem)
    metrics_lat = compute_metrics(y_true, y_lat)

    with open(join(path_to_save, "result.txt"), "w") as f:
        f.write(f"Tissue: {tissue}\n")
        f.write("\n=== FAISS on Semantic Embeddings ===\n")
        for k, v in metrics_sem.items():
            f.write(f"{k}: {v:.4f}\n")
        f.write("\n=== FAISS on Latent Embeddings ===\n")
        for k, v in metrics_lat.items():
            f.write(f"{k}: {v:.4f}\n")
        f.write(f"\nTime: {time.time() - start_t0:.2f} sec\n")

    with open(join(path_to_save, "time_log.txt"), "a") as f:
        f.write(f"AnnData + metrics time: {time.time() - t_adata0:.2f} sec\n")
        f.write(f"Total time: {time.time() - start_t0:.2f} sec\n")

    # Final cleanup
    release_cuda_memory()

    ## delete the checkpoint and the cache to save space
    import shutil
    shutil.rmtree(ckpt_dir, ignore_errors=True)
    if os.path.exists(cache_path):
        os.remove(cache_path)


# ----------------------------
# Batch runner
# ----------------------------

def discover_tissues(args: Args) -> List[str]:
    """Return a list of tissue directory names sorted by dataset size (ascending)."""
    tissues: List[str] = []
    sizes: List[int] = []

    for item in os.listdir(args.data_dir):
        tissue_path = join(args.data_dir, item)
        if not os.path.isdir(tissue_path):
            continue

        if args.use_parquet:
            train_parquets, test_parquets = get_all_parquets(tissue_path)
            if train_parquets and test_parquets:
                sz = sum(os.path.getsize(p) for p in train_parquets + test_parquets)
                tissues.append(item)
                sizes.append(sz)
        else:
            train_path = join(tissue_path, "train.h5ad")
            test_path = join(tissue_path, "test.h5ad")
            if os.path.exists(train_path) and os.path.exists(test_path):
                sz = os.path.getsize(train_path) + os.path.getsize(test_path)
                tissues.append(item)
                sizes.append(sz)

    tissues_sorted = [t for _, t in sorted(zip(sizes, tissues), key=lambda x: x[0])]
    return tissues_sorted


def job_fn(tissue: str, args: Args) -> None:
    try:
        run_tissue(tissue, args)
    except Exception as e:
        log_file = f"{args.save_root}/run_scLEAP.log"
        with open(log_file, "a") as f:
            f.write(f"Failed {tissue}: {e}\n")
        print(f"Failed {tissue}: {e}")


def run_all(args: Args) -> None:
    mp.set_start_method("spawn", force=True)
    warnings.filterwarnings("ignore")

    setup_environment_threads()
    set_cuda_visible(args.cuda_visible)
    pl.seed_everything(args.seed, workers=True)

    # Build a unique save root w/ hyperparams if the default is used
    
    # args.save_root = (
    #     f"{args.save_root}/scLEAP_full_hidden{args.hidden_dim}_s{args.s}_m{args.m1}_{args.m2}_{args.m3}"
    # )

    def tissue_has_data(tissue: str) -> bool:
        tissue_path = join(args.data_dir, tissue)
        if not os.path.isdir(tissue_path):
            return False
        if args.use_parquet:
            train_parquets, test_parquets = get_all_parquets(tissue_path)
            return bool(train_parquets and test_parquets)
        else:
            train_path = join(tissue_path, "train.h5ad")
            test_path = join(tissue_path, "test.h5ad")
            return os.path.exists(train_path) and os.path.exists(test_path)
    print(args.tissues)
    # Tissue selection: explicit names or discover all
    if args.tissues:
        selected, missing = [], []
        for t in args.tissues:
            if tissue_has_data(t):
                selected.append(t)
            else:
                missing.append(t)
        if missing:
            print("Warning: missing or invalid tissues:", ", ".join(missing))
        tissues = selected
    else:
        tissues = discover_tissues(args)
    print(tissues)

    if not tissues:
        print("No tissues to run. Exiting.")
        return

    # Skip already-completed tissues
    _, base_path = get_path_to_save(args, tissues[0])
    print(base_path)
    print(tissues)
    remain: List[str] = []
    for tissue in tissues:
        result_file = join(base_path, tissue, "result.txt")
        print(result_file)
        print(not os.path.exists(result_file))
        if not os.path.exists(result_file):
            remain.append(tissue)
    print(remain)
    print(f"Total of {len(remain)} tissues to run.")

    t0 = time.time()    
    if args.parallel and len(remain) > 200:
        print(f"Running in parallel with {args.workers} workers…")
        with mp.Pool(args.workers) as pool:
            pool.starmap(job_fn, [(t, args) for t in remain[:200]])
            pool.close()
            pool.join()
    else:
        for t in remain:
            job_fn(t, args)
    for t in remain[200:]:
        job_fn(t, args)

    total_time = time.time() - t0
    with open(f"{base_path}/args.txt", "w") as f:
        f.write(str(vars(args)))
        f.write(f"Total time: {total_time:.2f} seconds")
    print(f"Finished: {base_path} \n Total time: {total_time:.2f} seconds")


# ----------------------------
# CLI
# ----------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="scLEAP ablation/eval pipeline")

    # Data
    p.add_argument("--data-dir", required=True, help="Root directory of tissue subfolders")
    p.add_argument("--save-root", default="./results", help="Base directory to save outputs")
    p.add_argument("--use-parquet", action="store_true", help="Use parquet loaders instead of H5AD")
    p.add_argument("--external-data", action="store_true", help="If set, evaluate all classes (no in-train restriction)")
    p.add_argument("--ols-mappings-file", default=DEFAULT_OLS_MAPPINGS)
    p.add_argument("--ct-description-file", default=DEFAULT_CT_DESCRIPTIONS)
    p.add_argument("--tissues", nargs="+", help="Run only these tissue directory names (space-separated). Example: --tissues 'temporal cortex' liver")

    # Execution
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--max-epoch", type=int, default=200)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--parallel", action="store_true")
    p.add_argument("--gpu-id", type=int, default=0)
    p.add_argument("--cuda-visible", default=None, help="Comma-separated GPU ids to expose, e.g. '1,2'")

    # Model
    p.add_argument("--text-encoder-model", default="cambridgeltl/SapBERT-from-PubMedBERT-fulltext")
    p.add_argument("--hidden-dim", type=int, default=768)
    p.add_argument("--use-projection-head", action="store_true")
    p.add_argument("--n-cell-layers", type=int, default=5)
    p.add_argument("--n-text-layers-to-finetune", type=int, default=1)
    p.add_argument("--n-prompts", type=int, default=5)
    p.add_argument("--pool-type", default="mean", choices=["mean", "cls"])

    # Loss / logits
    p.add_argument("--s", type=float, default=30.0)
    p.add_argument("--m1", type=float, default=1.0)
    p.add_argument("--m2", type=float, default=0.3)
    p.add_argument("--m3", type=float, default=0.1)
    p.add_argument("--m4", type=float, default=1.5)
    p.add_argument("--f-gamma", type=float, default=1.3)
    p.add_argument("--l-ct", type=float, default=1.0)
    p.add_argument("--l-tt", type=float, default=0.1)
    p.add_argument("--l-contrastive", type=float, default=0.1)
    p.add_argument("--l-graph", type=float, default=0.1)
    p.add_argument("--graph-emb-npz", default=DEFAULT_GRAPH_EMB_NPZ)

    return p

def get_args_directly() -> Args:
    return Args(
        data_dir=str(REPO_ROOT / "data"),
        save_root="./results",
        use_parquet=True,
        external_data=False,
        ols_mappings_file=DEFAULT_OLS_MAPPINGS,
        ct_description_file=DEFAULT_CT_DESCRIPTIONS,
        seed=1,
        max_epoch=200,
        workers=8,
        parallel=False,
        gpu_id=6,
        cuda_visible='6',
        text_encoder_model="cambridgeltl/SapBERT-from-PubMedBERT-fulltext",
        hidden_dim=768,
        use_projection_head=False,
        n_cell_layers=5,
        n_text_layers_to_finetune=1,
        n_prompts=5,
        pool_type="mean",
        s=30.0,
        m1=1.0,
        m2=0.25,
        m3=0.1,
        m4=1.5,
        f_gamma=1.3,
        l_ct=1.0,
        l_tt=0.1,
        l_contrastive=0.1,
        l_graph=0.1,
        graph_emb_npz=DEFAULT_GRAPH_EMB_NPZ,
        tissues=['stomach'],
    )

def args_from_namespace(ns: argparse.Namespace) -> Args:
    return Args(
        data_dir=resolve_repo_path(ns.data_dir),
        save_root=resolve_repo_path(ns.save_root),
        use_parquet=bool(ns.use_parquet),
        external_data=bool(ns.external_data),
        ols_mappings_file=resolve_repo_path(ns.ols_mappings_file),
        ct_description_file=resolve_repo_path(ns.ct_description_file),
        seed=int(ns.seed),
        max_epoch=int(ns.max_epoch),
        workers=int(ns.workers),
        parallel=bool(ns.parallel),
        gpu_id=int(ns.gpu_id),
        cuda_visible=ns.cuda_visible,
        text_encoder_model=ns.text_encoder_model,
        hidden_dim=int(ns.hidden_dim),
        use_projection_head=bool(ns.use_projection_head),
        n_cell_layers=int(ns.n_cell_layers),
        n_text_layers_to_finetune=int(ns.n_text_layers_to_finetune),
        n_prompts=int(ns.n_prompts),
        pool_type=ns.pool_type,
        s=float(ns.s),
        m1=float(ns.m1),
        m2=float(ns.m2),
        m3=float(ns.m3),
        m4=float(ns.m4),
        f_gamma=float(ns.f_gamma),
        l_ct=float(ns.l_ct),
        l_tt=float(ns.l_tt),
        l_contrastive=float(ns.l_contrastive),
        l_graph=float(ns.l_graph),
        graph_emb_npz=resolve_repo_path(ns.graph_emb_npz),
        tissues=ns.tissues,
    )

#%%

if __name__ == "__main__":
    parser = build_parser()
    ns = parser.parse_args()
    args = args_from_namespace(ns)
    run_all(args)




""" 


## using parquet data
python analyses/intra_tissue_annotation/run_scLEAP.py --data-dir "data" \
    --save-root "./results" \
    --use-parquet \
    --cuda-visible 7 \
    --gpu-id 7 \
    --m1 1.0 \
    --m2 0.25 \
    --m3 0.1 \
    --tissues "colon"

## use h5ad
python analyses/intra_tissue_annotation/run_scLEAP.py --data-dir "data" \
    --tissues "left_lung" \
    --cuda-visible 3 \
    --gpu-id 3 \
    --max-epoch 200 \
    --m1 1.0 \
    --m2 0.25 \
    --m3 0.1
    
"""
