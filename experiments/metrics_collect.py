# -*- coding: utf-8 -*-
"""
Utilities to aggregate evaluation metrics for cell-type classification and clustering.

- build_classification_df(method_configs, tissues, num_workers=8) -> DataFrame
  Returns columns: ['tissue','method','accuracy','precision','recall','f1']

- build_clustering_df(clustering_method_configs) -> DataFrame
  Returns columns: ['tissue','method','nmi','ari']

- (Optional) load_scBERT(result_path) -> DataFrame
  Returns columns: ['tissue','method','accuracy','precision','recall','f1']
"""

import os
from multiprocessing import Pool

import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
from tqdm import tqdm


# ----------------------------- #
#      CLASSIFICATION PART      #
# ----------------------------- #

def get_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    return acc, precision, recall, f1


def _process_tissue_classification(args):
    """
    Worker: scan a tissue folder for result files, read predictions, compute metrics.
    Returns a dict with mean over all found result files for that tissue (stds dropped).
    """
    tissue, method, config = args
    result_file = config['result_file']
    ground_truth_key = config['ground_truth_key']
    pred_key = config['pred_key']
    result_dir = config['result_dir']
    idx2cal = config.get('idx2cal', False)

    try:
        tissue_path = os.path.join(result_dir, tissue)
        if not os.path.isdir(tissue_path):
            return None

        # collect candidate result files in tissue_path and its 1-level subdirs
        result_files = []
        direct_file = os.path.join(tissue_path, result_file)
        if os.path.isfile(direct_file):
            result_files.append(direct_file)
        for sub in os.listdir(tissue_path):
            sub_path = os.path.join(tissue_path, sub)
            file_path = os.path.join(sub_path, result_file)
            if os.path.isdir(sub_path) and os.path.isfile(file_path):
                result_files.append(file_path)

        if not result_files:
            return None

        all_metrics = []
        for file_path in result_files:
            try:
                result_adata = sc.read(file_path, backed='r')
                if idx2cal and 'idx2cal' in result_adata.obs:
                    result_adata = result_adata[result_adata.obs['idx2cal'].astype(bool)]
                y_true = result_adata.obs[ground_truth_key].values
                y_pred = result_adata.obs[pred_key].values
                metrics = get_metrics(y_true, y_pred)
                all_metrics.append(metrics)
            except Exception:
                continue

        if not all_metrics:
            return None

        avg = np.mean(np.array(all_metrics), axis=0)

        return {
            'tissue': tissue,
            'method': method,
            'accuracy': float(avg[0]),
            'precision': float(avg[1]),
            'recall': float(avg[2]),
            'f1': float(avg[3]),
        }
    except Exception:
        return None


def build_classification_df(method_configs, tissues, num_workers=8):
    """
    Evaluate all methods across tissues and return one concatenated DataFrame with columns:
    ['tissue','method','accuracy','precision','recall','f1']
    """
    method_results = []
    for method, config in method_configs.items():
        print(f"[Classification] Processing method: {method}")
        args_list = [(t, method, config) for t in tissues]
        with Pool(processes=num_workers) as pool:
            results = list(tqdm(pool.imap_unordered(_process_tissue_classification, args_list),
                                total=len(args_list)))
        results = [r for r in results if r is not None]
        if results:
            method_results.append(pd.DataFrame(results))

    if not method_results:
        return pd.DataFrame(columns=['tissue','method','accuracy','precision','recall','f1'])

    return pd.concat(method_results, ignore_index=True)


# ----------------------------- #
#        CLUSTERING PART        #
# ----------------------------- #

def load_clustering_scores(result_path, method, tissues=None):
    """
    Parse clustering scores from text files in {result_path}/{tissue}/.
    Supports several filename variants and line formats.

    Returns DataFrame with columns: ['tissue','method','nmi','ari']
    """
    if tissues is None:
        candidates = [d for d in os.listdir(result_path) if os.path.isdir(os.path.join(result_path, d))]
    else:
        candidates = tissues

    rows = []
    for tissue in candidates:
        tissue_path = os.path.join(result_path, tissue)
        if not os.path.isdir(tissue_path):
            continue

        # Choose file name based on method, otherwise try common alternatives
        result_file = None
        if method == 'scLMCT':
            seed1 = os.path.join(tissue_path, 'seed_1', 'clustering_result.txt')
            if os.path.exists(seed1):
                result_file = seed1
            else:
                fallback = os.path.join(tissue_path, 'clustering_result.txt')
                if os.path.exists(fallback):
                    result_file = fallback
        else:
            for fname in ['clustering_result.txt', 'result.txt', 'results.txt']:
                candidate = os.path.join(tissue_path, fname)
                if os.path.exists(candidate):
                    result_file = candidate
                    break

        if result_file is None or not os.path.exists(result_file):
            continue

        metrics = {
            'tissue': tissue,
            'method': method,
            'nmi': None,
            'ari': None,
        }

        with open(result_file, 'r') as f:
            for raw in f:
                line = raw.strip()
                # Robust parsing for different capitalizations/labels
                if line.lower().startswith("nmi:"):
                    try:
                        metrics['nmi'] = float(line.split(":")[1].strip())
                    except Exception:
                        pass
                elif line.lower().startswith("ari:"):
                    try:
                        metrics['ari'] = float(line.split(":")[1].strip())
                    except Exception:
                        pass

        rows.append(metrics)

    if not rows:
        return pd.DataFrame(columns=['tissue','method','nmi','ari'])

    df = pd.DataFrame(rows)
    return df[['tissue', 'method', 'nmi', 'ari']]


def build_clustering_df(clustering_method_configs):
    """
    Aggregate clustering metrics across methods into a single DataFrame.
    Expects: {'method_name': {'result_dir': '/path/to/dir'}, ...}
    Returns columns: ['tissue','method','nmi','ari']
    """
    dfs = []
    for method, cfg in clustering_method_configs.items():
        print(f"[Clustering] Processing method: {method}")
        df = load_clustering_scores(cfg['result_dir'], method)
        if not df.empty:
            dfs.append(df)
    if not dfs:
        return pd.DataFrame(columns=['tissue','method','nmi','ari'])
    return pd.concat(dfs, ignore_index=True)


# ----------------------------- #
#           EXAMPLES            #
# ----------------------------- #
if __name__ == "__main__":
    with open("./data/valid_tissues.txt") as f:
        tissues = f.read().splitlines()

        
    method_configs = {
        'scLMCT': {
            'result_dir': "./results/cell_type_annotation_results/scLMCT",
            'result_file': "pred.h5ad",
            'ground_truth_key': "true_labels_id",
            'pred_key': "pred_labels_faiss_lat_id",
            'idx2cal' : True

        },
        'scGPT': {
            'result_dir': "./results/cell_type_annotation_results/scGPT",
            'result_file': "embed_adata.h5ad",
            'ground_truth_key': "cell_type",
            'pred_key': "predictions",
            'idx2cal' : True

        },

        'scTab': {
            'result_dir': "./results/cell_type_annotation_results/scTab",
            'result_file': "pred.h5ad",
            'ground_truth_key': "cell_type",
            'pred_key': "predictions",
            'idx2cal' : True

        },
        
        'TOSICA' : {
            'result_dir': "./results/cell_type_annotation_results/TOSICA/",
            'result_file': "pred.h5ad",
            'ground_truth_key': "cell_type_id",
            'pred_key': "Prediction",
            'idx2cal' : True

        }
    
    }

    df_cls = build_classification_df(method_configs, tissues, num_workers=8)
    df_cls.to_csv("./results/cell_type_classification.csv", index=False)

    clustering_method_configs = {
        'scLMCT': {
            'result_dir': '/nfs/blanche/dungp/single-cell/cell_annotation/results/cell_type_annotation_results/scLMCT/',
        },
        'scanpy': {
            'result_dir': "./results/clustering_results/scanpy",
        },
        'Seurat': {
            'result_dir': "./results/clustering_results/Seurat",
        },
        'SHARP' : {
            'result_dir': "./results/clustering_results/SHARP",
        },
    }
    df_clu = build_clustering_df(clustering_method_configs)
    df_clu.to_csv("./results/cell_type_clustering.csv", index=False)