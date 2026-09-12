#!/usr/bin/env python3
import argparse
import os
import re
from pathlib import Path

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
import seaborn as sns
import torch
import torch.nn.functional as F
from scipy import sparse
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier
from tqdm import tqdm

from scleap.model import TrainWrapperCLIPStyle

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["BLAS_NUM_THREADS"] = "1"

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_DIR = REPO_ROOT / "data" / "zeroshot-data"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "zero_shot_annotation"


def resolve_repo_path(path_str: str) -> Path:
    path = Path(path_str).expanduser()
    if path.is_absolute():
        return path
    return path.resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run zero-shot prediction on one or more H5AD datasets.")
    parser.add_argument("--ckpt-path", required=True, help="Path to a scLEAP checkpoint.")
    parser.add_argument("--datasets", nargs="+", required=True, help="Dataset basenames without .h5ad.")
    parser.add_argument("--data-dir", default=str(DEFAULT_INPUT_DIR), help="Directory containing <dataset>.h5ad files.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Directory for per-dataset outputs.")
    parser.add_argument("--label-col", default="cell_type", help="Observation column containing labels.")
    return parser.parse_args()


def process_new_celltypes(model, cell_types, label_prompts=None):
    int_to_ct = {i: ct for i, ct in enumerate(cell_types)}

    model.eval()

    if label_prompts is None:
        prompt_templates = [
            "A single-cell transcriptome from a {label}.",
            "This is the gene expression profile of a {label}.",
            "This cell is classified as a {label} cell.",
            "This cell type is called {label} cell.",
            "Semantic annotation for this cell: {label} cell.",
            "This is a {label} cell.",
            "This is a cell of type {label}.",
        ]
        cell_types_local = list(int_to_ct.values())
        cleaned_cell_types = [
            re.sub(r"\bcell\b", "", ct, flags=re.IGNORECASE).replace("_", " ").strip()
            for ct in cell_types_local
        ]
        label_prompts = {
            i: [template.format(label=label) for template in prompt_templates]
            for i, label in enumerate(cleaned_cell_types)
        }
    else:
        label_prompts = {i: label_prompts[ct] for i, ct in int_to_ct.items()}

    model.label_prompts = label_prompts
    model.prompt_token_batches = {
        class_id: model.text_encoder_wrapper.tokenizer(
            prompts,
            padding=True,
            truncation=True,
            max_length=256,
            return_tensors="pt",
        )
        for class_id, prompts in label_prompts.items()
    }

    model.on_train_start()
    model.text_encoder_wrapper.cache_path = "./temp_cache.pt"
    if os.path.exists(model.text_encoder_wrapper.cache_path):
        os.remove(model.text_encoder_wrapper.cache_path)

    model.prompt_layer10_cache = {}
    model.setup()


def extract_embeddings(adata, model, batch_size=1024):
    model.eval()

    if sparse.issparse(adata.X):
        x = adata.X.toarray()
    else:
        x = adata.X

    num_batches = x.shape[0] // batch_size + 1
    embeddings = []

    for i in range(num_batches):
        start = i * batch_size
        end = min((i + 1) * batch_size, x.shape[0])
        if start >= end:
            continue

        batch = torch.tensor(x[start:end]).float().cuda()
        with torch.no_grad():
            embedding = model(batch)
            embeddings.append(embedding.cpu())

    return torch.cat(embeddings, dim=0)


def plot_embeddings(
    model,
    adata,
    embeddings,
    int_to_ct,
    cell_types,
    label_col="cell_type",
    save_path=None,
    tsne_save_path=None,
):
    labels = []
    text_embeddings = []

    with torch.no_grad():
        for _ in range(3):
            model.n_prompts = -1
            lb, t_embeddings = model.get_class_text_centers(
                labels=torch.LongTensor(np.arange(len(cell_types))),
                only_centers=True,
            )
            labels.append(lb)
            text_embeddings.append(t_embeddings)

    labels = torch.cat(labels, dim=0)
    text_embeddings = torch.cat(text_embeddings, dim=0)

    text_embeddings = F.normalize(text_embeddings, dim=-1)
    labels = labels.detach().cpu().numpy()
    text_embeddings = text_embeddings.detach().cpu().numpy()
    labels = [int_to_ct[label] for label in labels]

    adata.obs["cell_type_ols"] = adata.obs[label_col].astype(str)

    valid_labels = set(adata.obs["cell_type_ols"].unique())
    text_mask_keep = [label in valid_labels for label in labels]
    text_embeddings = text_embeddings[text_mask_keep]
    text_labels = [labels[i] for i, keep in enumerate(text_mask_keep) if keep]

    x_expr = embeddings
    x_text = text_embeddings
    x_stacked = np.vstack([x_expr, x_text])

    adata_full = adata.copy()
    adata_full = adata_full.concatenate(
        sc.AnnData(x_text),
        batch_key="type",
        batch_categories=["cell", "text"],
        index_unique=None,
    )
    adata_full.obsm["X_stacked"] = x_stacked

    cell_labels = adata.obs["cell_type_ols"].tolist()
    adata_full.obs["cell_type_plot"] = cell_labels + text_labels

    all_labels = adata_full.obs["cell_type_plot"].unique()
    palette = sns.color_palette("tab20", n_colors=len(all_labels))
    label_color_map = dict(zip(all_labels, palette))

    sc.pp.neighbors(adata_full, use_rep="X_stacked")
    sc.tl.umap(adata_full)

    fig, ax = plt.subplots(figsize=(8, 6))
    umap = adata_full.obsm["X_umap"]
    cell_mask = adata_full.obs["type"] == "cell"
    text_mask = adata_full.obs["type"] == "text"

    sc.pl.umap(
        adata_full[cell_mask],
        color="cell_type_plot",
        palette=label_color_map,
        ax=ax,
        show=False,
    )

    text_labels_series = adata_full.obs.loc[text_mask, "cell_type_plot"]
    for label in sorted(set(text_labels_series)):
        idx = (text_labels_series == label).values
        coords = umap[text_mask][idx]
        ax.scatter(
            coords[:, 0],
            coords[:, 1],
            color=label_color_map[label],
            marker="^",
            edgecolor="black",
            linewidth=0.8,
            s=80,
            label=f"text: {label}",
        )

    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5), fontsize=7)
    plt.title("UMAP: Cells vs Text Embeddings")
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()

    sc.tl.tsne(adata_full, use_rep="X_stacked", n_pcs=None)

    fig, ax = plt.subplots(figsize=(8, 6))
    tsne = adata_full.obsm["X_tsne"]
    cell_mask = adata_full.obs["type"] == "cell"
    text_mask = adata_full.obs["type"] == "text"

    cell_df = pd.DataFrame(
        {
            "tsne1": tsne[cell_mask, 0],
            "tsne2": tsne[cell_mask, 1],
            "cell_type_plot": adata_full.obs.loc[cell_mask, "cell_type_plot"].values,
        }
    )

    for label in all_labels:
        sub = cell_df[cell_df["cell_type_plot"] == label]
        if len(sub) > 0:
            ax.scatter(
                sub["tsne1"],
                sub["tsne2"],
                s=6,
                alpha=0.7,
                color=label_color_map[label],
                label=label,
            )

    text_labels_series = adata_full.obs.loc[text_mask, "cell_type_plot"]
    for label in sorted(set(text_labels_series)):
        idx = (text_labels_series == label).values
        coords = tsne[text_mask][idx]
        ax.scatter(
            coords[:, 0],
            coords[:, 1],
            color=label_color_map[label],
            marker="^",
            edgecolor="black",
            linewidth=0.8,
            s=80,
            label=f"text: {label}",
        )

    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5), fontsize=7)
    plt.title("t-SNE: Cells vs Text Embeddings")
    plt.tight_layout()
    if tsne_save_path is not None:
        plt.savefig(tsne_save_path, dpi=300, bbox_inches="tight")
    plt.show()

    return adata_full


def impute_and_reorder_genes(
    adata: ad.AnnData,
    target_genes: list[str],
    dtype=None,
) -> ad.AnnData:
    n_cells = adata.n_obs
    target_genes = list(target_genes)
    target_index = {g: i for i, g in enumerate(target_genes)}

    src_genes = adata.var_names.tolist()
    src_index = {g: i for i, g in enumerate(src_genes)}
    common_genes = [g for g in target_genes if g in src_index]

    if dtype is None:
        dtype = adata.X.dtype

    if sp.issparse(adata.X):
        x_out = sp.lil_matrix((n_cells, len(target_genes)), dtype=dtype)
        x_src = adata.X.tocsr()
    else:
        x_out = np.zeros((n_cells, len(target_genes)), dtype=dtype)
        x_src = adata.X

    for g in common_genes:
        src_i = src_index[g]
        tgt_i = target_index[g]
        x_out[:, tgt_i] = x_src[:, src_i]

    if sp.issparse(x_out):
        x_out = x_out.tocsr()

    var_out = pd.DataFrame(index=target_genes)
    var_out["imputed"] = [g not in src_index for g in target_genes]

    return ad.AnnData(
        X=x_out,
        obs=adata.obs.copy(),
        var=var_out,
        uns=adata.uns.copy(),
    )


def main():
    args = parse_args()
    ckpt_path = resolve_repo_path(args.ckpt_path)
    data_dir = resolve_repo_path(args.data_dir)
    output_dir = resolve_repo_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = TrainWrapperCLIPStyle.load_from_checkpoint(str(ckpt_path))

    for dts_name in tqdm(args.datasets):
        try:
            label_col = args.label_col
            adata = sc.read(data_dir / f"{dts_name}.h5ad")

            adata.obs[label_col] = adata.obs[label_col].astype(str).str.lower()

            if adata.X.max() > 20:
                sc.pp.normalize_total(adata, target_sum=1e4)
                sc.pp.log1p(adata)

            cell_types = adata.obs[label_col].unique().tolist()
            int_to_ct = {i: ct for i, ct in enumerate(cell_types)}
            process_new_celltypes(model, cell_types)

            embeddings = extract_embeddings(adata, model).cpu().numpy()
            embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
            print(embeddings.shape)

            dts_save_path = output_dir / dts_name
            dts_save_path.mkdir(parents=True, exist_ok=True)

            adata_full = plot_embeddings(
                model,
                adata,
                embeddings,
                int_to_ct,
                cell_types,
                label_col=label_col,
                save_path=dts_save_path / f"{dts_name}_umap.png",
                tsne_save_path=dts_save_path / f"{dts_name}_tsne.png",
            )

            labels = []
            text_embeddings = []
            with torch.no_grad():
                for _ in range(10):
                    model.n_prompts = 5
                    lb, t_embeddings = model.get_class_text_centers(
                        labels=torch.LongTensor(np.arange(len(cell_types))),
                        only_centers=True,
                    )
                    labels.append(lb)
                    text_embeddings.append(t_embeddings)

            labels = torch.cat(labels, dim=0)
            text_embeddings = torch.cat(text_embeddings, dim=0)

            text_embeddings = F.normalize(text_embeddings, dim=-1)
            labels = labels.detach().cpu().numpy()
            text_embeddings = text_embeddings.detach().cpu().numpy()

            adata.obs["cell_type_ols"] = adata.obs[label_col].astype(str)

            clf = KNeighborsClassifier(n_neighbors=5, metric="cosine", weights="distance")
            clf.fit(text_embeddings, labels)

            pred_embeddings = extract_embeddings(adata, model).cpu().numpy()
            pred_embeddings = pred_embeddings / np.linalg.norm(pred_embeddings, axis=1, keepdims=True)

            predictions = clf.predict(pred_embeddings)
            predictions = np.array([int_to_ct[label] for label in predictions])

            y_true = adata.obs["cell_type_ols"].values
            y_pred = predictions
            report = classification_report(y_true, y_pred, output_dict=False)
            print(report)

            pred_adata = sc.AnnData(X=pred_embeddings, obs=adata.obs.copy())
            pred_adata.obs["predicted_cell_type"] = y_pred
            pred_adata.write_h5ad(dts_save_path / f"{dts_name}_predictions.h5ad")
            print(f"Predictions saved to {dts_save_path / f'{dts_name}_predictions.h5ad'}")

            umap_df = pd.DataFrame(adata_full.obsm["X_umap"], columns=["umap1", "umap2"])
            umap_df.index = adata_full.obs.index
            umap_df = pd.concat([umap_df, adata_full.obs], axis=1)
            umap_df.to_csv(dts_save_path / f"{dts_name}_umap.csv", index=True)

            tsne_df = pd.DataFrame(adata_full.obsm["X_tsne"], columns=["tsne1", "tsne2"])
            tsne_df.index = adata_full.obs.index
            tsne_df = pd.concat([tsne_df, adata_full.obs], axis=1)
            tsne_df.to_csv(dts_save_path / f"{dts_name}_tsne.csv", index=True)

            with open(dts_save_path / "results.txt", "w") as f:
                f.write(report)

        except Exception as e:
            print(f"Error processing {dts_name}: {e}")
            continue


if __name__ == "__main__":
    main()
