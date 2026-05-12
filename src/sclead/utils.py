import os
import json
import random
import numpy as np
from datetime import datetime
from typing import Dict, Any, Optional
import json
from importlib.resources import files

import torch
import torch.nn as nn
import torch.nn.functional as f

import h5py
from sklearn.cluster import KMeans

try:
    from cuml.manifold.umap import UMAP
except:
    from umap import UMAP
# Random seed functions
DEFAULT_RANDOM_SEED = 1

def seedBasic(seed=DEFAULT_RANDOM_SEED):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

def seedTorch(seed=DEFAULT_RANDOM_SEED):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def seedEverything(seed=DEFAULT_RANDOM_SEED):
    seedBasic(seed)
    seedTorch(seed)





import gc
import scanpy as sc
import torch
import numpy as np
import time
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score, adjusted_mutual_info_score, silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
def get_latent_representations(data, model, batch_size=256, device="cuda"):
    dataset = TensorDataset(torch.tensor(data, dtype=torch.float32))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    all_latents = []
    model.eval()
    model.to(device)
    with torch.no_grad():
        for batch in dataloader:
            inputs = batch[0].to(device)
            latents = model(inputs)
            all_latents.append(latents.cpu().numpy())
    return np.vstack(all_latents)

import matplotlib.cm as cm
import matplotlib.colors as colors


def merge_clusters(latent, cluster_labels, threshold = 0.8):
    unique_clusters = np.unique(cluster_labels)
    cluster_centers = {}
    
    # Compute cluster centers
    for cluster in unique_clusters:
        cluster_mask = cluster_labels == cluster
        cluster_centers[cluster] = latent[cluster_mask].mean(axis=0)
    
    # Convert to array for similarity calculation
    clusters = list(cluster_centers.keys())
    centers = np.array([cluster_centers[c] for c in clusters])
    
    # Compute cosine similarity between cluster centers
    similarity_matrix = cosine_similarity(centers)
    print(similarity_matrix)
    # Merge clusters based on similarity threshold
    merged_labels = cluster_labels.copy()
    merged_map = {c: c for c in clusters}
    # threshold = compute_threshold(similarity_matrix)
    

    print(threshold)
    for i in range(len(clusters)):
        for j in range(i + 1, len(clusters)):
            if similarity_matrix[i, j] > threshold:
                target_cluster = merged_map[clusters[j]]
                source_cluster = merged_map[clusters[i]]
                
                # Merge cluster j into cluster i
                merged_labels[merged_labels == target_cluster] = source_cluster
                merged_map[clusters[j]] = source_cluster
    
    return merged_labels


def get_cluster_results(train_latent=None, test_latent=None, train_labels=None, test_labels=None,
                        use_pca=False, cal_umap=False, cal_silhouette=True):
    start = time.time()

    # Build full dataset for UMAP + plotting
    if train_latent is not None and train_labels is not None:
        latents = np.concatenate([train_latent, test_latent])
        labels = np.concatenate([train_labels, test_labels])
        data_types = ['train'] * len(train_latent) + ['test'] * len(test_latent)
        test_indices = np.arange(len(train_latent), len(latents))
    else:
        latents = test_latent
        labels = test_labels
        data_types = ['test'] * len(test_latent)
        test_indices = np.arange(len(latents))

    adata = sc.AnnData(latents)
    adata.obs['cell_type'] = labels
    adata.obs['data_type'] = data_types

    if cal_umap:
        umap = UMAP(n_neighbors=15)
        adata.obsm['X_umap'] = umap.fit_transform(latents)
    print("UMAP done")

    # Clustering only on test data
    test_adata = sc.AnnData(test_latent)
    sc.pp.neighbors(test_adata, use_rep='X')

    try:
        import cudf
        import cugraph

        connectivities = test_adata.obsp['connectivities']
        sources, targets = connectivities.nonzero()
        weights = connectivities.data

        df = cudf.DataFrame()
        df['source'] = sources
        df['destination'] = targets
        df['weight'] = weights

        G = cugraph.Graph()
        G.from_cudf_edgelist(df, source='source', destination='destination', edge_attr='weight')
        leiden_parts, _ = cugraph.leiden(G, resolution=0.8)
        leiden_assignments = leiden_parts.to_pandas().sort_values('vertex')['partition'].values
        test_leiden = leiden_assignments.astype(str)
    except (ImportError, ModuleNotFoundError):
        print("cuGraph not available, falling back to CPU Leiden")
        sc.tl.leiden(test_adata, key_added='leiden', resolution=0.8)
        test_leiden = test_adata.obs['leiden'].values

    cluster_label_key = 'leiden'
    merged_cluster_key = 'merged_leiden_pca'
    print("Leiden clustering on test set done")

    # Merge clusters on test latent + test_leiden
    merged_cluster_label = merge_clusters(test_latent, test_leiden)
    print("Merged clusters done")

    # Store labels in the full adata
    leiden_full = np.full(len(adata), 'NA', dtype=object)
    merged_full = np.full(len(adata), 'NA', dtype=object)
    leiden_full[test_indices] = test_leiden
    merged_full[test_indices] = merged_cluster_label.astype(str)

    adata.obs[cluster_label_key] = leiden_full
    adata.obs[merged_cluster_key] = merged_full

    # Evaluation
    true_label = np.array(labels)[test_indices]
    pred_label = merged_cluster_label

    nmi = normalized_mutual_info_score(true_label, pred_label)
    ari = adjusted_rand_score(true_label, pred_label)
    ami = adjusted_mutual_info_score(true_label, pred_label)

    silhouette = silhouette_cluster = -1
    if cal_silhouette and cal_umap:
        print("calculating silhouette")
        try:
            test_umap = adata.obsm['X_umap'][test_indices]
            silhouette = silhouette_score(test_umap, true_label)
            silhouette_cluster = silhouette_score(test_umap, pred_label)
        except Exception as e:
            print("Silhouette error:", e)

    exec_time = time.time() - start
    print("All done")
    return nmi, ari, silhouette, silhouette_cluster, ami, exec_time, adata, cluster_label_key, merged_cluster_key


import time
import numpy as np
import scanpy as sc
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score, adjusted_mutual_info_score, silhouette_score
from umap import UMAP
#%%
def cluster_and_plot_umap(test_latent, test_labels, cal_umap=True, cal_silhouette=True, merge_threshold=0.7):
    start = time.time()

    # Build AnnData for test data
    adata = sc.AnnData(test_latent)
    adata.obs['cell_type'] = test_labels

    # UMAP for visualization
    if cal_umap:
        umap = UMAP(n_neighbors=15)
        adata.obsm['X_umap'] = umap.fit_transform(test_latent)
        print("UMAP done")

    # Clustering (Leiden) on test data
    sc.pp.neighbors(adata, use_rep='X')
    try:
        import cudf
        import cugraph

        connectivities = adata.obsp['connectivities']
        sources, targets = connectivities.nonzero()
        weights = connectivities.data

        df = cudf.DataFrame()
        df['source'] = sources
        df['destination'] = targets
        df['weight'] = weights

        G = cugraph.Graph()
        G.from_cudf_edgelist(df, source='source', destination='destination', edge_attr='weight')
        leiden_parts, _ = cugraph.leiden(G, resolution=0.8)
        leiden_assignments = leiden_parts.to_pandas().sort_values('vertex')['partition'].values
        cluster_labels = leiden_assignments.astype(str)
    except (ImportError, ModuleNotFoundError):
        print("cuGraph not available, falling back to CPU Leiden")
        sc.tl.leiden(adata, key_added='leiden', resolution=0.8)
        cluster_labels = adata.obs['leiden'].values

    

    cluster_label_key = 'leiden'
    merged_cluster_key = 'merged_leiden'
    print("Leiden clustering on test set done")

    # Merge clusters on test latent + test_leiden
    adata.obs[cluster_label_key] = cluster_labels
    merged_cluster_label = merge_clusters(test_latent, cluster_labels, threshold = merge_threshold)
    adata.obs[merged_cluster_key] = merged_cluster_label
    print("Merged clusters done")

    # Metrics
    nmi = normalized_mutual_info_score(test_labels, merged_cluster_label)
    ari = adjusted_rand_score(test_labels, merged_cluster_label)
    ami = adjusted_mutual_info_score(test_labels, merged_cluster_label)

    silhouette, silhouette_cluster = -1, -1
    if cal_silhouette and cal_umap:
        print("calculating silhouette")
        try:
            test_umap = adata.obsm['X_umap']
            silhouette = silhouette_score(test_umap, merged_cluster_label)
            silhouette_cluster = silhouette_score(test_umap, merged_cluster_label)
        except Exception as e:
            print("Silhouette error:", e)

    exec_time = time.time() - start
    print("All done")
    return nmi, ari, silhouette, silhouette_cluster, ami, exec_time, adata

# Example usage:
# nmi, ari, silhouette, silhouette_cluster, ami, exec_time, adata = cluster_and_plot_umap(test_latent, test_labels)


import matplotlib.cm as cm
import matplotlib.colors as colors
def plot_umap(adata, original_cluster_key, merged_cluster_key, use_pca=False):
    print("Start plotting")
    
    # First Figure: Combined, Train, Test UMAP by cell type
    fig1, axes1 = plt.subplots(1, 3, figsize=(21, 6))
    for idx, group in enumerate(["all", "train", "test"]):
        if group == "all":
            sub = adata
            title = "All Cells"
        else:
            if group not in adata.obs['data_type'].values:
                continue  # skip if group not present
            sub = adata[adata.obs['data_type'] == group].copy()
            title = f"{group.capitalize()} Cells"

        sc.pl.umap(sub, color='cell_type', ax=axes1[idx], show=False)
        axes1[idx].set_title(title)
        axes1[idx].set_aspect("equal")

    plt.tight_layout()

    # Second Figure: Only Test cells by clustering
    fig2, axes2 = plt.subplots(1, 3, figsize=(21, 6))
    test_adata = adata[adata.obs['data_type'] == 'test'].copy()

    sc.pl.umap(test_adata, color='cell_type', ax=axes2[0], show=False)
    axes2[0].set_title("Test Cell Types")

    sc.pl.umap(test_adata, color=original_cluster_key, ax=axes2[1], show=False)
    axes2[1].set_title(f"Test {original_cluster_key} {'(PCA)' if use_pca else ''}")

    test_adata.obs['merged_clusters_plot'] = test_adata.obs[merged_cluster_key].astype('category')
    unique_clusters = test_adata.obs['merged_clusters_plot'].cat.categories
    cluster_palette = sc.pl.palettes.default_20[:len(unique_clusters)]

    sc.pl.umap(test_adata, color='merged_clusters_plot', ax=axes2[2], show=False, palette=cluster_palette)
    axes2[2].set_title(f"Test {merged_cluster_key} {'(PCA)' if use_pca else ''}")

    plt.tight_layout()
    print("Plot done")
    return fig1, fig2


import seaborn as sns
import pandas as pd

def plot_umap_predictions(pred_adata, pred_key, save_path, save_name):
    # Compute UMAP
    pred_adata = pred_adata[pred_adata.obs['idx2cal'] == True].copy()
    sc.pp.neighbors(pred_adata, use_rep="X")
    sc.tl.umap(pred_adata)

    # Create subplots
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    # Plot true labels
    sc.pl.umap(
        pred_adata, 
        color="true_labels", 
        ax=axs[0], 
        show=False, 
        title="True Cell Types", 
        legend_loc='on data'
    )

    # Plot predicted labels
    sc.pl.umap(
        pred_adata, 
        color=pred_key, 
        ax=axs[1], 
        show=False, 
        title="Predicted Cell Types", 
        legend_loc='on data'
    )

    # Save the combined plot
    fig.tight_layout()
    fig.savefig(os.path.join(save_path, save_name), dpi=300)
    plt.close(fig)

    
    
    # Extract UMAP and labels
    umap_df = pd.DataFrame(
        pred_adata.obsm["X_umap"],
        columns=["UMAP1", "UMAP2"]
    )
    umap_df["label"] = pred_adata.obs[pred_key].values

    # Set up plot
    fig, ax = plt.subplots(figsize=(10, 8))  # square figure
    sns.scatterplot(
        data=umap_df,
        x="UMAP1",
        y="UMAP2",
        hue="label",
        palette="tab20",  # or a custom palette
        ax=ax,
        s=10,
        linewidth=0
    )

    ax.set_title("Predicted Cell Types")
    ax.set_aspect("equal", adjustable="box")
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

    fig.savefig(os.path.join(save_path, "pred_only_test_umap.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)




def predict_with_distance_weights(index, query_vectors, labels, k, return_distances=False, return_neighbor_labels=False):
    distances, indices = index.search(query_vectors, k)
    similarities = distances + 1.0  # Convert distances to similarities

    unique_labels = np.unique(labels)
    class_counts = {label: np.sum(labels == label) for label in unique_labels}
    total_samples = len(labels)
    class_weights = {label: total_samples / (len(unique_labels) * count) for label, count in class_counts.items()}

    predictions = []
    all_distances = []
    all_neighbor_labels = []

    for i in range(query_vectors.shape[0]):
        neighbor_indices = indices[i]
        neighbor_labels = labels[neighbor_indices]
        neighbor_weights = similarities[i]

        weighted_votes = {label: 0.0 for label in unique_labels}
        for label, weight in zip(neighbor_labels, neighbor_weights):
            adjusted_weight = weight * class_weights[label]
            weighted_votes[label] += adjusted_weight

        max_vote_label = max(weighted_votes.items(), key=lambda x: x[1])[0]
        predictions.append(max_vote_label)

        all_distances.append(distances[i])
        all_neighbor_labels.append(neighbor_labels)

    outputs = [np.array(predictions)]
    if return_distances:
        outputs.append(np.array(all_distances))
    if return_neighbor_labels:
        outputs.append(np.stack(all_neighbor_labels))  # shape: (n_queries, k)

    return tuple(outputs)




import numpy as np
import matplotlib.pyplot as plt
import umap
from sklearn.preprocessing import LabelEncoder
def plot_embeddings(exprs_embeddings, labels, text_embeddings, text_labels, save_path):
    import numpy as np
    import matplotlib.pyplot as plt
    import umap
    from sklearn.preprocessing import LabelEncoder

    # Combine and encode labels
    all_labels = np.concatenate([labels, text_labels])
    label_encoder = LabelEncoder()
    label_colors = label_encoder.fit_transform(all_labels)
    unique_labels = label_encoder.classes_
    num_classes = len(unique_labels)

    # Apply UMAP
    all_embeddings = np.vstack([exprs_embeddings, text_embeddings])
    sources = np.array(['expr'] * len(exprs_embeddings) + ['text'] * len(text_embeddings))
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.3, random_state=42)
    reduced = reducer.fit_transform(all_embeddings)

    # Assign colors
    import matplotlib.cm as cm
    colormap = cm.get_cmap('tab10', num_classes)
    color_dict = {label: colormap(i) for i, label in enumerate(unique_labels)}

    # Plot
    plt.figure(figsize=(10, 8))

    for i, label in enumerate(unique_labels):
        # Get indices for each label
        expr_idx = np.where((sources == 'expr') & (all_labels == label))[0]
        text_idx = np.where((sources == 'text') & (all_labels == label))[0]

        # Plot expression embeddings as dots
        plt.scatter(reduced[expr_idx, 0], reduced[expr_idx, 1],
                    label=label, color=color_dict[label], s=20, alpha=0.6)

        # Plot text embeddings as stars
        plt.scatter(reduced[text_idx, 0], reduced[text_idx, 1],
                    marker='*', color=color_dict[label], edgecolor='black', s=200)

    plt.legend(title="Cell Type", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.title("UMAP of Expression and Text Embeddings")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()

def get_semantics_labels(train_label_dict, ols_mappings=None):
    """
    Return:
        semantics_labels: dict
            key   = integer class id
            value = list of text prompts/descriptions
    """

    simple_templates = [
        "A single-cell transcriptome from a {label} cell.",
        "This is the gene expression profile of a {label} cell.",
        "Cell type: {label} cell.",
        "Based on its gene expression, this cell is a {label} cell.",
        "A biologically annotated cell profile: {label} cell.",
        "An scRNA-seq profile labeled as: {label} cell.",
        "This cell's identity is: {label} cell.",
        "This cell is classified as a {label} cell.",
        "Transcriptomic identity: {label} cell.",
        "This vector encodes the functional signature of a {label} cell.",
    ]

    def make_default_prompts(label):
        return [template.format(label=label) for template in simple_templates]

    try:
        desc_db = load_cell_types_info()

        semantics_labels = {}

        if ols_mappings is None:
            for label, class_id in train_label_dict.items():
                if label in desc_db:
                    semantics_labels[class_id] = desc_db[label]
                else:
                    semantics_labels[class_id] = make_default_prompts(label)

        else:
            for label, class_id in train_label_dict.items():
                # label is usually ontology ID, e.g. CL:0000625
                # ols_mappings may map ontology ID -> ontology name or vice versa
                mapped_label = ols_mappings.get(label, label)

                if mapped_label in desc_db:
                    semantics_labels[class_id] = desc_db[mapped_label]
                elif label in desc_db:
                    semantics_labels[class_id] = desc_db[label]
                else:
                    semantics_labels[class_id] = make_default_prompts(label)

        semantics_labels = {
            k: semantics_labels[k]
            for k in sorted(semantics_labels.keys())
        }

    except Exception as e:
        print("Using fallback prompts because:", repr(e))

        semantics_labels = {}
        for label, class_id in train_label_dict.items():
            semantics_labels[class_id] = make_default_prompts(label)

        semantics_labels = {
            k: semantics_labels[k]
            for k in sorted(semantics_labels.keys())
        }

    return semantics_labels

def load_cell_types_info() -> dict:
    path = files("sclead.ct_descriptions") / "cell_types_info.json"
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def load_cell_types_mapping() -> dict:
    path = files("sclead.ct_descriptions") / "cell_name_to_ols_id.json"
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)