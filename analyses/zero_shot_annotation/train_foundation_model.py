#!/usr/bin/env python3
"""
Train a single scLEAP (CLIP-style) model on one parquet dataset folder.

Expected folder structure (one dataset):
  DATA_DIR/
    train_parquets/*.parquet
    train_ontology_to_int.json            (required; ontology_id -> int)

Parquet schema assumptions:
  - Each row has:
      * "X"            : list/array of float features
      * "ontology_int" : int64 label id (matches values in train_ontology_to_int.json)

Cell type descriptions / prompts:
  - ct_description_file may point to a JSON mapping ontology_id (e.g. "CL:0000540") to:
      * list[str] prompts, OR
      * dict containing "prompts"/"description"/etc (common cases handled)
  - If a label is missing, we fall back to templates.

Assumes these project-local modules exist on PYTHONPATH:
  - model.TrainWrapperCLIPStyle
  - gpu_parquet_window_loader_v1.build_loader
"""

from __future__ import annotations
import sys
# ----------------------------
# Thread + CUDA env (set early)
# ----------------------------
import os

THREAD_ENV_VARS = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "BLAS_NUM_THREADS": "1",
}
for k, v in THREAD_ENV_VARS.items():
    os.environ.setdefault(k, v)

# ----------------------------
# Standard libs
# ----------------------------
import argparse
import gc
import json
import time
import warnings
from dataclasses import dataclass
from os.path import join
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pytorch_lightning as pl
import torch

# Project-local deps
from scleap.model import TrainWrapperCLIPStyle
from scleap.gpu_parquet_dataset import build_loader
from scleap.utils import load_cell_types_info, load_cell_types_mapping

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CT_DESCRIPTIONS = str(REPO_ROOT / "src" / "scleap" / "ct_descriptions" / "cell_types_info.json")


# ----------------------------
# Args
# ----------------------------
@dataclass
class Args:
    data_dir: str
    save_dir: str
    ct_description_file: str = DEFAULT_CT_DESCRIPTIONS

    # execution
    cuda_visible: str | None = None   # e.g. "3"
    gpu_id: int = 0                  # index into visible GPUs (usually 0 if cuda_visible is set)

    # parquet loader
    window_rows: int = 16384
    rmm_pool: str = "off"
    batch_size: int | None = None
    drop_last: bool = True

    # graph embeddings (optional)
    graph_emb_npz: str | None = None  # NPZ with keys: names, embeddings
    graph_l2_normalize: bool = True   # L2-normalize graph embeddings before use
    graph_max_norm: float | None = None  # Optional clip: rows with norm>max are scaled down

    # model
    text_encoder_model: str = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"
    hidden_dim: int = 768
    n_cell_layers: int = 5
    n_text_layers_to_finetune: int = 1
    n_prompts: int = 5
    pool_type: str = "mean"  # "mean" or "cls"
    use_projection_head: bool = False

    # loss
    s: float = 30.0
    m1: float = 1.0
    m2: float = 0.25
    m3: float = 0.1
    m4: float = 0.8
    f_gamma: float = 1.3
    l_ct: float = 0.5
    l_tt: float = 0.1
    l_contrastive: float = 0.1
    l_graph: float = 0.1  # keep simple

    # train
    seed: int = 1
    max_epoch: int = 50
    lr: float = 1e-4
    weight_decay: float = 1e-4
    gradient_clip_val: float = 1.0


# ----------------------------
# Helpers
# ----------------------------
def set_cuda_visible(devs: str | None) -> None:
    if devs is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = devs


def resolve_repo_path(path_str: str | None) -> str | None:
    if path_str is None:
        return None
    path = Path(path_str).expanduser()
    if path.is_absolute():
        return str(path)
    return str((REPO_ROOT / path).resolve())


def get_train_parquets(data_dir: str) -> List[str]:
    train_dir = join(data_dir, "train_parquets")
    if not os.path.isdir(train_dir):
        return []
    return [join(train_dir, f) for f in sorted(os.listdir(train_dir)) if f.endswith(".parquet")]


def determine_batch_size_parquet(n_parts: int) -> int:
    if n_parts <= 1:
        return 128
    if n_parts <= 4:
        return 2048
    return 8192


def load_train_label_dict(data_dir: str) -> Dict[str, int]:
    """
    train_ontology_to_int.json is expected to be:
      { "CL:0000....": 0, ... }
    """
    path = join(data_dir, "train_ontology_to_int.json")
    with open(path, "r") as f:
        d = json.load(f)
    return {str(k).strip().lower(): int(v) for k, v in d.items()}


def _prompts_from_desc_obj(obj) -> List[str] | None:
    if obj is None:
        return None
    if isinstance(obj, list) and all(isinstance(x, str) for x in obj):
        return obj
    if isinstance(obj, dict):
        for key in ("prompts", "prompt"):
            if key in obj and isinstance(obj[key], list) and all(isinstance(x, str) for x in obj[key]):
                return obj[key]
        for key in ("description", "text", "def", "definition"):
            if key in obj and isinstance(obj[key], str) and obj[key].strip():
                s = obj[key].strip()
                return [
                    f"A single-cell transcriptome consistent with: {s}",
                    f"Cell identity description: {s}",
                    f"Transcriptomic profile matches: {s}",
                    f"Biological annotation: {s}",
                    f"This cell type can be described as: {s}",
                ]
    return None


def build_semantics_labels(ct_description_file: str, train_label_dict: Dict[str, int], n_prompts: int) -> Dict[int, List[str]]:
    templates = [
        "A single-cell transcriptome from a {label} cell.",
        "This is the gene expression profile of a {label} cell.",
        "Cell type: {label} cell.",
        "Based on its gene expression, this cell is a {label} cell.",
        "A biologically annotated cell profile: {label} cell.",
    ]

    raw = {}
    # try:
    #     with open(ct_description_file, "r") as f:
    #         raw = json.load(f)
    # except Exception as e:
    #     warnings.warn(f"Could not read ct_description_file={ct_description_file} ({e}). Using template prompts only.")
    #     raw = {}

    print(f"Loading cell type descriptions")
    raw = load_cell_types_info()
    print("successfully loaded cell type descriptions, first 3 entries:")
    for i, (k, v) in enumerate(list(raw.items())[:3]):
        print(f"  {i}: {k} -> {v}")

    norm_desc = {str(k).strip().lower(): v for k, v in (raw or {}).items()}

    semantics: Dict[int, List[str]] = {}
    for ont_id, class_id in train_label_dict.items():
        desc_obj = norm_desc.get(ont_id, None)
        prompts = _prompts_from_desc_obj(desc_obj)

        if prompts is None:
            base = ont_id.replace("_", " ")
            prompts = [t.format(label=base) for t in templates]

        if len(prompts) >= n_prompts:
            prompts = prompts[:n_prompts]
        else:
            while len(prompts) < n_prompts:
                prompts.append(prompts[-1])

        semantics[class_id] = prompts

    return {k: semantics[k] for k in sorted(semantics.keys())}


# ----------------------------
# Graph embeddings helpers (optional)
# ----------------------------

def _l2_normalize_rows(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.maximum(norms, eps)
    return x / norms


def _clip_row_norms(x: np.ndarray, max_norm: float, eps: float = 1e-12) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    scale = np.minimum(1.0, max_norm / np.maximum(norms, eps))
    return x * scale


def load_graph_npz(npz_path: str) -> Tuple[Dict[str, int], np.ndarray]:
    """
    NPZ is expected to contain:
      - names: shape [N] (strings; e.g., 'cl:0000057' or 'CL:0000057')
      - embeddings: shape [N, D] float/float32

    Returns:
      name_to_idx (lowercased/stripped) and embeddings array (float32).
    """
    data = np.load(npz_path, allow_pickle=True)
    print(f"Loaded graph NPZ from {npz_path} with keys: {list(data.keys())}")
    if 'names' not in data or 'embeddings' not in data:
        raise KeyError(f"{npz_path} must contain keys: 'names' and 'embeddings'")
    names = data['names']
    emb = np.asarray(data['embeddings'], dtype=np.float32)
    name_to_idx = {str(n).strip().lower(): i for i, n in enumerate(names)}
    return name_to_idx, emb


def build_graph_embeddings_for_train_labels(
    train_label_dict: Dict[str, int],
    name_to_idx: Dict[str, int],
    embeddings: np.ndarray,
    *,
    l2_normalize: bool = True,
    max_norm: float | None = None,
) -> Tuple[np.ndarray, int]:
    """
    Build graph_embeddings aligned to training class ids:
      graph_embeddings[class_id] = embeddings[name_to_idx[ontology_id]]

    Missing ontology ids remain zeros (same behavior as prior scripts).

    Returns:
      (graph_embeddings, n_matched)
    """
    emb = np.asarray(embeddings, dtype=np.float32)
    if l2_normalize:
        emb = _l2_normalize_rows(emb)
    if max_norm is not None:
        emb = _clip_row_norms(emb, float(max_norm))

    graph_embeddings = np.zeros((len(train_label_dict), emb.shape[1]), dtype=np.float32)
    matched = 0
    for ont_id, class_id in train_label_dict.items():
        key = str(ont_id).strip().lower()
        idx = name_to_idx.get(key, None)
        if idx is not None:
            graph_embeddings[int(class_id)] = emb[idx]
            matched += 1
    return graph_embeddings, matched


def release_cuda_memory() -> None:
    try:
        torch.cuda.synchronize()
    except Exception:
        pass
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass
    gc.collect()


# ----------------------------
# Main train
# ----------------------------
def main(args: Args) -> None:
    set_cuda_visible(args.cuda_visible)
    pl.seed_everything(args.seed, workers=True)
    warnings.filterwarnings("ignore")

    os.makedirs(args.save_dir, exist_ok=True)
    ckpt_dir = join(args.save_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    cache_path = join(args.save_dir, "cached_prompt_embeddings.pt")

    train_parquets = get_train_parquets(args.data_dir)
    if not train_parquets:
        raise FileNotFoundError(f"No train parquets found under: {join(args.data_dir, 'train_parquets')}")

    train_label_dict = load_train_label_dict(args.data_dir)  # ontology_id -> class_id
    semantics_labels = build_semantics_labels(args.ct_description_file, train_label_dict, n_prompts=args.n_prompts)

    # ----------------------------
    # Graph embeddings (optional)
    # ----------------------------
    graph_embeddings = None
    if args.graph_emb_npz:
        try:
            name_to_idx, emb = load_graph_npz(args.graph_emb_npz)
            graph_embeddings, n_matched = build_graph_embeddings_for_train_labels(
                train_label_dict,
                name_to_idx,
                emb,
                l2_normalize=bool(args.graph_l2_normalize),
                max_norm=args.graph_max_norm,
            )
            if not np.isfinite(graph_embeddings).all():
                raise ValueError('graph_embeddings contains NaN/Inf')
            pct = 100.0 * n_matched / max(1, len(train_label_dict))
            print(
                f"Graph embeddings enabled: {args.graph_emb_npz} | matched {n_matched}/{len(train_label_dict)} ({pct:.1f}%) | shape={tuple(graph_embeddings.shape)}"
            )
            if n_matched == 0:
                warnings.warn('0 graph embeddings matched training labels. Disabling graph embeddings.')
                graph_embeddings = None
        except Exception as e:
            warnings.warn(f"Failed to load/align graph embeddings from {args.graph_emb_npz}: {e}. Disabling graph embeddings.")
            graph_embeddings = None

    print(graph_embeddings is not None and "Using graph embeddings." or "Not using graph embeddings.")

    columns = ["X", "ontology_int"]
    list_columns = ["X"]
    label_columns = ["ontology_int"]
    casts = {"ontology_int": "int64"}

    batch_size = args.batch_size if args.batch_size is not None else determine_batch_size_parquet(len(train_parquets))

    train_loader = build_loader(
        files=train_parquets,
        columns=columns,
        list_columns=list_columns,
        label_columns=label_columns,
        casts=casts,
        batch_size=batch_size,
        window_rows=args.window_rows,
        shuffle=True,
        part_shuffle=True,
        drop_last=args.drop_last,
        seed=args.seed,
        device_id=args.gpu_id,
        rmm_pool=args.rmm_pool,
    )

    # infer input dim
    first_batch = next(iter(train_loader))
    x0, y0 = first_batch
    if isinstance(x0, dict):
        d_input = x0["X"].shape[-1]
    else:
        d_input = x0.shape[-1]

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
        lr=args.lr,
        weight_decay=args.weight_decay,
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
        graph_embeddings=graph_embeddings,
    )

    callbacks = [
        pl.callbacks.ModelCheckpoint(
            dirpath=ckpt_dir,
            monitor="train_loss",
            save_top_k=5,
            mode="min",
            every_n_epochs=1,
        ),
        pl.callbacks.EarlyStopping(
            monitor="train_loss",
            patience=5,
            min_delta=1e-3,
            mode="min",
        ),
    ]

    trainer = pl.Trainer(
        max_epochs=args.max_epoch,
        accelerator="gpu",
        devices=1,
        callbacks=callbacks,
        gradient_clip_val=args.gradient_clip_val,
        deterministic=True,
        enable_progress_bar=True,
        log_every_n_steps=50,
    )

    t0 = time.time()
    trainer.fit(model, train_loader)
    train_time = time.time() - t0

    best_path = callbacks[0].best_model_path if hasattr(callbacks[0], "best_model_path") else ""
    if best_path and os.path.isfile(best_path):
        model = TrainWrapperCLIPStyle.load_from_checkpoint(best_path)
        final_ckpt = join(args.save_dir, "best.ckpt")
        try:
            import shutil
            shutil.copy2(best_path, final_ckpt)
        except Exception:
            pass

    with open(join(args.save_dir, "run_info.json"), "w") as f:
        json.dump(
            {
                "data_dir": args.data_dir,
                "save_dir": args.save_dir,
                "n_train_parquets": len(train_parquets),
                "n_classes": len(train_label_dict),
                "input_dim": int(d_input),
                "best_checkpoint": best_path,
                "train_time_sec": float(train_time),
                "args": vars(args),
            },
            f,
            indent=2,
        )

    release_cuda_memory()
    print(f"\nDone. Saved to: {args.save_dir}")


# ----------------------------
# CLI
# ----------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("Train scLEAP on a single parquet dataset folder")

    p.add_argument("--data-dir", required=True, help="Dataset folder containing train_parquets/ and train_ontology_to_int.json")
    p.add_argument("--save-dir", required=True, help="Output folder for checkpoints + run_info.json")
    p.add_argument("--ct-description-file", default=DEFAULT_CT_DESCRIPTIONS, help="Optional JSON mapping ontology_id -> prompts/description")

    p.add_argument("--cuda-visible", default=None, help="Comma-separated GPUs to expose, e.g. '3'")
    p.add_argument("--gpu-id", type=int, default=0, help="Index into visible GPUs (usually 0)")

    # model
    p.add_argument("--text-encoder-model", default="cambridgeltl/SapBERT-from-PubMedBERT-fulltext")
    p.add_argument("--hidden-dim", type=int, default=768)
    p.add_argument("--n-cell-layers", type=int, default=5)
    p.add_argument("--n-text-layers-to-finetune", type=int, default=1)
    p.add_argument("--n-prompts", type=int, default=5)
    p.add_argument("--pool-type", default="mean", choices=["mean", "cls"])
    p.add_argument("--use-projection-head", action="store_true")

    # graph embeddings (optional)
    p.add_argument("--graph-emb-npz", default=None, help="Path to NPZ with keys {names, embeddings}. If set, enables graph embeddings.")
    p.add_argument("--graph-no-l2-normalize", action="store_true", help="Disable L2 normalization of graph embeddings (default: normalize).")
    p.add_argument("--graph-max-norm", type=float, default=None, help="Optional: clip each graph embedding row norm to this value after (optional) normalization.")

    # loss
    p.add_argument("--s", type=float, default=30.0)
    p.add_argument("--m1", type=float, default=1.0)
    p.add_argument("--m2", type=float, default=0.25)
    p.add_argument("--m3", type=float, default=0.1)
    p.add_argument("--m4", type=float, default=0.8)
    p.add_argument("--f-gamma", type=float, default=1.3)
    p.add_argument("--l-ct", type=float, default=0.5)
    p.add_argument("--l-tt", type=float, default=0.1)
    p.add_argument("--l-contrastive", type=float, default=0.1)
    p.add_argument("--l-graph", type=float, default=0.1)

    # train
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--max-epoch", type=int, default=50)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--gradient-clip-val", type=float, default=1.0)

    # parquet loader
    p.add_argument("--batch-size", type=int, default=None, help="Override loader batch size")
    p.add_argument("--window-rows", type=int, default=16384)
    p.add_argument("--rmm-pool", default="off", choices=["off", "pool", "managed"])
    p.add_argument("--drop-last", action="store_true", help="Drop last incomplete batch (recommended for training)")

    return p


def args_from_ns(ns: argparse.Namespace) -> Args:
    return Args(
        data_dir=resolve_repo_path(ns.data_dir),
        save_dir=resolve_repo_path(ns.save_dir),
        ct_description_file=resolve_repo_path(ns.ct_description_file),
        cuda_visible=ns.cuda_visible,
        gpu_id=ns.gpu_id,
        text_encoder_model=ns.text_encoder_model,
        hidden_dim=ns.hidden_dim,
        n_cell_layers=ns.n_cell_layers,
        n_text_layers_to_finetune=ns.n_text_layers_to_finetune,
        n_prompts=ns.n_prompts,
        pool_type=ns.pool_type,
        use_projection_head=bool(ns.use_projection_head),
        s=ns.s,
        m1=ns.m1,
        m2=ns.m2,
        m3=ns.m3,
        m4=ns.m4,
        f_gamma=ns.f_gamma,
        l_ct=ns.l_ct,
        l_tt=ns.l_tt,
        l_contrastive=ns.l_contrastive,
        l_graph=ns.l_graph,
        seed=ns.seed,
        max_epoch=ns.max_epoch,
        lr=ns.lr,
        weight_decay=ns.weight_decay,
        gradient_clip_val=ns.gradient_clip_val,
        batch_size=ns.batch_size,
        window_rows=ns.window_rows,
        rmm_pool=ns.rmm_pool,
        drop_last=bool(ns.drop_last),
        graph_emb_npz=resolve_repo_path(ns.graph_emb_npz),
        graph_l2_normalize=not bool(ns.graph_no_l2_normalize),
        graph_max_norm=ns.graph_max_norm,
    )


if __name__ == "__main__":
    parser = build_parser()
    ns = parser.parse_args()
    args = args_from_ns(ns)
    main(args)

"""
Example:

python train_foundation_model.py \
  --data-dir "data/<dataset_name>" \
  --save-dir "outputs/foundation_model" \
  --cuda-visible 7 \
  --gpu-id 0 \
  --max-epoch 200 \
  --m1 1 --m2 0.25 --m3 0.1 \
  --drop-last \
  --n-text-layers-to-finetune 1\
  --n-cell-layers 5 \
  --l-graph 0.000 \
  --l-ct 1.0 \
    --l-tt 0.0 \
    --l-contrastive 0.0 \
    --pool-type cls \
    --graph-emb-npz "data/graph_embeddings/cl_poincare_embeddings.npz"
    
"""
