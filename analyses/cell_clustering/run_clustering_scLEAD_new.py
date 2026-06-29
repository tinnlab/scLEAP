#!/usr/bin/env python3
"""
scLEAD embedding + clustering benchmark (Scanpy-style), per tissue.

What this script does
---------------------
For each tissue (folder) under --data_dir:
  1) Load <tissue>/test.h5ad
  2) Deterministically subsample to --max_cells (optional, shared manifest)
  3) Normalize + log1p (optional safety check)
  4) Extract embeddings using ONE checkpoint model (TrainWrapperCLIPStyle)
  5) kNN graph + Leiden clustering on embeddings (cuGraph if available, else CPU)
  6) Optional centroid-cosine cluster merging
  7) Compute NMI / ARI / AMI (+ optional silhouette if UMAP computed)
  8) Save:
       - <result_dir>/<tissue>/pred.h5ad   (embeddings + clustering columns)
       - <result_dir>/<tissue>/clusters.csv
       - <result_dir>/<tissue>/results.txt

Notes / assumptions
-------------------
- The model is loaded ONCE from --ckpt_path.
- This script assumes your TrainWrapperCLIPStyle can encode cell features from adata.X.
  By default, it tries these in order:
      a) model.encode(X_tensor) if exists
      b) model.encoder(X_tensor) if exists
      c) model(X_tensor) otherwise
  If your wrapper uses a different method name, adjust _encode_batch().
- Data feeding is batched (default --batch_size 1024).
- Default does NOT compute UMAP (use --cal_umap to enable).
"""

import os

# ---- hard limit BLAS threads (important when using multiprocessing) ----
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["BLAS_NUM_THREADS"] = "1"

import argparse
import hashlib
import time
import random
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import scanpy as sc

from sklearn.metrics import (
    normalized_mutual_info_score,
    adjusted_rand_score,
    adjusted_mutual_info_score,
    silhouette_score,
)
from sklearn.metrics.pairwise import cosine_similarity

# model / torch
import torch

# Your project-local model
from sclead.model import TrainWrapperCLIPStyle

try:
    from umap import UMAP  # optional
except Exception:
    UMAP = None


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TISSUE_CSV = str(REPO_ROOT / "data" / "tissue_cell_counts.csv")


# ----------------------------
# Utilities
# ----------------------------
def set_random_seed(seed: int):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_repo_path(path_str: str) -> str:
    path = Path(path_str).expanduser()
    if path.is_absolute():
        return str(path)
    return str((REPO_ROOT / path).resolve())


def _sha1_hex(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


def get_or_create_subsample_manifest_from_cell_ids(
    *,
    cell_ids: List[str],
    dataset: str,
    subsample_dir: str,
    max_cells: int,
    seed: int,
) -> List[str]:
    """
    Deterministically select cells by sorting cell_ids using SHA1(f"{seed}:{cell_id}")
    and taking first max_cells. Save/read manifest so other methods can reuse exact cell set.
    """
    os.makedirs(subsample_dir, exist_ok=True)
    manifest_path = os.path.join(subsample_dir, f"{dataset}__max{max_cells}__seed{seed}.txt")

    if os.path.exists(manifest_path):
        ids = [ln.strip() for ln in open(manifest_path, "r") if ln.strip()]
        if len(ids) == 0:
            raise RuntimeError(f"Manifest exists but is empty: {manifest_path}")
        return ids

    if max_cells <= 0 or len(cell_ids) <= max_cells:
        chosen = list(cell_ids)
    else:
        keys = [_sha1_hex(f"{seed}:{cid}") for cid in cell_ids]
        order = np.argsort(np.array(keys, dtype=object))
        chosen = [cell_ids[i] for i in order[:max_cells]]

    with open(manifest_path, "w") as f:
        for cid in chosen:
            f.write(cid + "\n")
    return chosen


def maybe_normalize_log1p(adata: sc.AnnData, *, target_sum: float = 1e4) -> None:
    """
    Conservative normalization: only normalize/log1p if values look like raw counts.
    Your previous heuristic was max > 20; keep that behavior.
    """
    try:
        mx = adata.X.max()
    except Exception:
        mx = np.asarray(adata.X).max()
    if mx > 20:
        sc.pp.normalize_total(adata, target_sum=target_sum)
        sc.pp.log1p(adata)


def l2_normalize_rows(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.clip(norms, eps, None)


# ----------------------------
# Embedding extraction
# ----------------------------
def _encode_batch(model, xb: torch.Tensor) -> torch.Tensor:
    """
    Tries common patterns for your wrapper.
    Adjust here if your TrainWrapperCLIPStyle uses a different method.
    """
    if hasattr(model, "encode") and callable(getattr(model, "encode")):
        return model.encode(xb)
    if hasattr(model, "encoder") and callable(getattr(model, "encoder")):
        return model.encoder(xb)
    return model(xb)


@torch.no_grad()
def extract_embeddings(
    adata: sc.AnnData,
    *,
    model,
    device: str,
    batch_size: int,
    normalize_embedding: bool,
) -> np.ndarray:
    """
    Extract embeddings for adata.X using the loaded model. Returns float32 numpy array (n_cells, d).
    """
    model.eval()
    if hasattr(model, "to"):
        model.to(device)

    X = adata.X
    # Make sure we have a dense float32 array for torch
    if hasattr(X, "toarray"):
        X = X.toarray()
    X = np.asarray(X, dtype=np.float32)

    n = X.shape[0]
    outs = []

    for i in range(0, n, batch_size):
        xb = torch.from_numpy(X[i : i + batch_size]).to(device)
        zb = _encode_batch(model, xb)
        zb = zb.detach().float().cpu().numpy()
        outs.append(zb)

    Z = np.concatenate(outs, axis=0).astype(np.float32, copy=False)
    if normalize_embedding:
        Z = l2_normalize_rows(Z)
    return Z


# ----------------------------
# Clustering + merging
# ----------------------------
def merge_clusters_by_centroid_cosine(latent: np.ndarray, cluster_labels: np.ndarray, *, threshold: float) -> np.ndarray:
    labels = np.asarray(cluster_labels).astype(str)
    uniq = np.unique(labels)

    centers = []
    for c in uniq:
        m = labels == c
        centers.append(latent[m].mean(axis=0))
    centers = np.asarray(centers)

    sim = cosine_similarity(centers)

    merged = labels.copy()
    merged_map = {c: c for c in uniq}

    for i in range(len(uniq)):
        for j in range(i + 1, len(uniq)):
            if sim[i, j] > threshold:
                target = merged_map[uniq[j]]
                source = merged_map[uniq[i]]
                merged[merged == target] = source
                merged_map[uniq[j]] = source

    return merged


def leiden_on_embeddings(
    X: np.ndarray,
    *,
    n_neighbors: int,
    resolution: float,
    use_cugraph_if_available: bool = True,
) -> Tuple[np.ndarray, str]:
    ad = sc.AnnData(X)
    sc.pp.neighbors(ad, n_neighbors=n_neighbors, use_rep="X")

    if use_cugraph_if_available:
        try:
            import cudf
            import cugraph

            conn = ad.obsp["connectivities"]
            sources, targets = conn.nonzero()
            weights = conn.data

            df = cudf.DataFrame({"source": sources, "destination": targets, "weight": weights})
            G = cugraph.Graph()
            G.from_cudf_edgelist(df, source="source", destination="destination", edge_attr="weight")

            parts, _ = cugraph.leiden(G, resolution=resolution)
            labels = parts.to_pandas().sort_values("vertex")["partition"].to_numpy().astype(str)
            return labels, "cugraph"
        except Exception:
            pass

    sc.tl.leiden(ad, key_added="leiden", resolution=resolution)
    return ad.obs["leiden"].to_numpy().astype(str), "scanpy"


def maybe_umap(X: np.ndarray, *, n_neighbors: int, seed: int) -> np.ndarray:
    if UMAP is None:
        raise RuntimeError("UMAP requested but umap-learn is not installed.")
    um = UMAP(n_neighbors=n_neighbors, random_state=seed)
    return um.fit_transform(X)


# ----------------------------
# Tissue iteration
# ----------------------------
def list_tissues_with_test(data_dir: str) -> List[str]:
    out = []
    for item in os.listdir(data_dir):
        tp = os.path.join(data_dir, item)
        if not os.path.isdir(tp):
            continue
        if os.path.exists(os.path.join(tp, "test.h5ad")):
            out.append(item)
    return sorted(out)


def run_one_tissue(
    tissue: str,
    *,
    model,
    data_dir: str,
    result_dir: str,
    subsample_dir: str,
    max_cells: int,
    seed: int,
    label_key: str,
    n_neighbors: int,
    resolution: float,
    merge: bool,
    merge_threshold: float,
    cal_umap: bool,
    cal_silhouette: bool,
    normalize_before_model: bool,
    normalize_embedding: bool,
    batch_size: int,
    device: str,
    use_cugraph_if_available: bool,
) -> None:
    out_dir = os.path.join(result_dir, tissue)
    os.makedirs(out_dir, exist_ok=True)

    results_txt = os.path.join(out_dir, "results.txt")
    if os.path.exists(results_txt):
        print(f"[SKIP] {tissue}")
        return

    test_path = os.path.join(data_dir, tissue, "test.h5ad")

    start = time.time()
    try:
        print(f"[RUN] tissue={tissue}")

        # 1) Load test.h5ad
        adata = sc.read_h5ad(test_path)

        if label_key not in adata.obs.columns:
            raise RuntimeError(f"Missing label_key='{label_key}' in {test_path}. Available: {list(adata.obs.columns)[:20]}")

        # 2) Deterministic subsample (optional; always writes/uses manifest)
        cell_ids = adata.obs_names.to_list()
        chosen = get_or_create_subsample_manifest_from_cell_ids(
            cell_ids=cell_ids,
            dataset=tissue,
            subsample_dir=subsample_dir,
            max_cells=max_cells,
            seed=seed,
        )
        chosen_present = [cid for cid in chosen if cid in adata.obs_names]
        if len(chosen_present) == 0:
            raise RuntimeError(f"Chosen manifest cells not found in adata.obs_names for tissue={tissue}")
        was_subsampled = len(chosen_present) < adata.n_obs
        if was_subsampled:
            adata = adata[chosen_present].copy()

        # 3) Normalize/log1p before feeding to model (as requested)
        if normalize_before_model:
            maybe_normalize_log1p(adata, target_sum=1e4)

        # 4) Extract embeddings with checkpoint model
        Z = extract_embeddings(
            adata,
            model=model,
            device=device,
            batch_size=batch_size,
            normalize_embedding=normalize_embedding,
        )

        # Save embeddings into pred AnnData (same structure as your other benchmark outputs)
        pred = sc.AnnData(Z)
        pred.obs = adata.obs.copy()
        pred.obs_names = adata.obs_names.copy()
        pred.var_names = [f"emb_{i}" for i in range(Z.shape[1])]

        # Optional UMAP for viz / silhouette
        X_umap = None
        if cal_umap:
            X_umap = maybe_umap(Z, n_neighbors=n_neighbors, seed=seed)
            pred.obsm["X_umap"] = X_umap

        # 5) Leiden clustering on embeddings
        leiden_labels, backend = leiden_on_embeddings(
            Z,
            n_neighbors=n_neighbors,
            resolution=resolution,
            use_cugraph_if_available=use_cugraph_if_available,
        )
        pred.obs["leiden"] = leiden_labels

        # 6) Merge clusters (optional)
        final_labels = leiden_labels
        if merge:
            merged = merge_clusters_by_centroid_cosine(Z, leiden_labels, threshold=merge_threshold)
            pred.obs["merged_leiden"] = merged
            final_labels = merged
        else:
            pred.obs["merged_leiden"] = leiden_labels

        # 7) Metrics
        y_true = pred.obs[label_key].astype(str).to_numpy()

        nmi = normalized_mutual_info_score(y_true, final_labels)
        ari = adjusted_rand_score(y_true, final_labels)
        ami = adjusted_mutual_info_score(y_true, final_labels)

        sil_true = -1
        sil_cluster = -1
        if cal_silhouette and cal_umap and X_umap is not None:
            try:
                sil_cluster = silhouette_score(X_umap, final_labels)
                sil_true = silhouette_score(X_umap, y_true)
            except Exception as e:
                print(f"[WARN] silhouette failed for {tissue}: {e}")

        elapsed = time.time() - start

        # 8) Write outputs
        pred_path = os.path.join(out_dir, "pred.h5ad")
        pred.write_h5ad(pred_path)

        cluster_df = pd.DataFrame(
            {
                "cell": pred.obs_names.to_list(),
                "true_label": y_true,
                "leiden": pred.obs["leiden"].astype(str).to_numpy(),
                "merged_leiden": pred.obs["merged_leiden"].astype(str).to_numpy(),
            }
        )
        cluster_df.to_csv(os.path.join(out_dir, "clusters.csv"), index=False)

        with open(results_txt, "w") as f:
            f.write(f"Tissue: {tissue}\n")
            f.write("Method: scLEAD_checkpoint_embeddings+Leiden\n")
            f.write(f"ckpt_path: {os.path.abspath(str(args_ckpt_path_placeholder))}\n")  # replaced in main()
            f.write(f"Leiden_backend: {backend}\n")
            f.write(f"n_neighbors: {n_neighbors}\n")
            f.write(f"resolution: {resolution}\n")
            f.write(f"merge: {merge}\n")
            f.write(f"merge_threshold: {merge_threshold}\n")
            f.write(f"NMI: {nmi}\n")
            f.write(f"ARI: {ari}\n")
            f.write(f"AMI: {ami}\n")
            f.write(f"silhouette_true: {sil_true}\n")
            f.write(f"silhouette_cluster: {sil_cluster}\n")
            f.write(f"Time: {elapsed}\n")
            f.write(f"Subsampled: {was_subsampled}\n")
            f.write(f"Final_n_cells: {pred.n_obs}\n")
            f.write(f"Max_cells: {max_cells}\n")
            f.write(f"Seed: {seed}\n")
            f.write(f"normalize_before_model: {normalize_before_model}\n")
            f.write(f"normalize_embedding: {normalize_embedding}\n")
            f.write(f"batch_size: {batch_size}\n")
            f.write(f"device: {device}\n")

        print(f"[OK] {tissue} -> {out_dir}")

    except Exception as e:
        with open(os.path.join(out_dir, "error.log"), "w") as f:
            f.write(f"Error processing tissue {tissue}: {repr(e)}\n")
        print(f"[ERR] {tissue}: {e}")


def main():
    ap = argparse.ArgumentParser(description="Per-tissue scLEAD checkpoint embeddings -> Leiden(+merge) clustering benchmark.")
    ap.add_argument("--ckpt_path", type=str, required=True, help="Checkpoint path for TrainWrapperCLIPStyle")
    ap.add_argument("--data_dir", type=str, required=True, help="Directory containing <tissue>/test.h5ad")
    ap.add_argument("--result_dir", type=str, required=True, help="Output directory (per-tissue subfolders)")

    ap.add_argument("--label_key", type=str, default="cell_type", help="obs column in test.h5ad for ground-truth labels")
    ap.add_argument("--seed", type=int, default=1)

    # subsampling (shared manifest)
    ap.add_argument("--max_cells", type=int, default=300000, help="Deterministic subsample size (<=0 means no subsample)")
    ap.add_argument("--subsample_dir", type=str, default=None, help="Where to store subsample manifests (default: <result_dir>/_subsamples)")

    # model / embedding
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--batch_size", type=int, default=1024)
    ap.add_argument("--normalize_before_model", action="store_true", help="Apply normalize_total+log1p before embedding (default OFF)")
    ap.add_argument("--normalize_embedding", action="store_true", help="L2-normalize embeddings row-wise (default OFF)")

    # clustering params
    ap.add_argument("--n_neighbors", type=int, default=15)
    ap.add_argument("--resolution", type=float, default=0.8)
    ap.add_argument("--no_cugraph", action="store_true")

    # merge options
    ap.add_argument("--merge", action="store_true")
    ap.add_argument("--merge_threshold", type=float, default=0.8)

    # optional UMAP / silhouette
    ap.add_argument("--cal_umap", action="store_true", help="Compute UMAP and store in pred.h5ad (default OFF)")
    ap.add_argument("--cal_silhouette", action="store_true", help="Compute silhouette on UMAP (requires --cal_umap)")

    ap.add_argument("--tissue", type=str, default=None, help="Run a single tissue (subfolder name)")
    ap.add_argument(
        "--tissue_csv",
        type=str,
        default=DEFAULT_TISSUE_CSV,
        help="CSV listing tissues in a 'tissue' column when --tissue is not set",
    )

    args = ap.parse_args()
    args.ckpt_path = resolve_repo_path(args.ckpt_path)
    args.data_dir = resolve_repo_path(args.data_dir)
    args.result_dir = resolve_repo_path(args.result_dir)
    args.tissue_csv = resolve_repo_path(args.tissue_csv)

    set_random_seed(args.seed)
    os.makedirs(args.result_dir, exist_ok=True)

    subsample_dir = args.subsample_dir or os.path.join(args.result_dir, "_subsamples")
    os.makedirs(subsample_dir, exist_ok=True)

    # Load model ONCE
    print(f"[MODEL] loading checkpoint: {args.ckpt_path}")
    model = TrainWrapperCLIPStyle.load_from_checkpoint(args.ckpt_path, map_location=args.device)
    model.eval()

    # Record run provenance
    run_info = os.path.join(args.result_dir, "_run_info_sclmct_ckpt_leiden.txt")
    if not os.path.exists(run_info):
        with open(run_info, "w") as f:
            f.write(f"ckpt_path: {args.ckpt_path}\n")
            f.write(f"data_dir: {args.data_dir}\n")
            f.write(f"result_dir: {args.result_dir}\n")
            f.write(f"label_key: {args.label_key}\n")
            f.write(f"seed: {args.seed}\n")
            f.write(f"max_cells: {args.max_cells}\n")
            f.write(f"n_neighbors: {args.n_neighbors}\n")
            f.write(f"resolution: {args.resolution}\n")
            f.write(f"merge: {args.merge}\n")
            f.write(f"merge_threshold: {args.merge_threshold}\n")
            f.write(f"cal_umap: {args.cal_umap}\n")
            f.write(f"cal_silhouette: {args.cal_silhouette}\n")
            f.write(f"device: {args.device}\n")
            f.write(f"batch_size: {args.batch_size}\n")
            f.write(f"normalize_before_model: {args.normalize_before_model}\n")
            f.write(f"normalize_embedding: {args.normalize_embedding}\n")
            f.write(f"cugraph_enabled: {not args.no_cugraph}\n")

    # Tissues
    if args.tissue:
        tissues = [args.tissue]
    else:
        tissues = pd.read_csv(args.tissue_csv)["tissue"].tolist()

    print(f"Tissues to process: {len(tissues)}")

    # Small hack to pass ckpt_path into results writer without global mutable state in signatures
    global args_ckpt_path_placeholder
    args_ckpt_path_placeholder = args.ckpt_path

    for tissue in tissues:
        test_path = os.path.join(args.data_dir, tissue, "test.h5ad")
        if not os.path.exists(test_path):
            print(f"[SKIP] {tissue}: missing test.h5ad")
            continue

        run_one_tissue(
            tissue,
            model=model,
            data_dir=args.data_dir,
            result_dir=args.result_dir,
            subsample_dir=subsample_dir,
            max_cells=args.max_cells,
            seed=args.seed,
            label_key=args.label_key,
            n_neighbors=args.n_neighbors,
            resolution=args.resolution,
            merge=args.merge,
            merge_threshold=args.merge_threshold,
            cal_umap=args.cal_umap,
            cal_silhouette=args.cal_silhouette,
            normalize_before_model=args.normalize_before_model,
            normalize_embedding=args.normalize_embedding,
            batch_size=args.batch_size,
            device=args.device,
            use_cugraph_if_available=(not args.no_cugraph),
        )

    print("DONE")


if __name__ == "__main__":
    main()

"""
Example:

CUDA_VISIBLE_DEVICES=7 python run_clustering_scLEAD_new.py \
  --ckpt_path ../checkpoint.ckpt \
  --data_dir data \
  --result_dir ./clustering_results \
    --tissue_csv data/tissue_cell_counts.csv \
  --label_key cell_type \
  --seed 1 \
  --max_cells 30000000 \
  --normalize_before_model \
  --normalize_embedding \
  --merge --merge_threshold 0.95

Default behavior:
- No UMAP unless you add --cal_umap.
- No silhouette unless you add --cal_umap --cal_silhouette.
"""
