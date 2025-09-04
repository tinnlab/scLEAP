import os

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ["BLAS_NUM_THREADS"] = "1"

import numpy as np
import pandas as pd
import time
import scanpy as sc
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score, silhouette_score, adjusted_mutual_info_score
import random
import multiprocessing as mp

base_path = "./data/cellxgene"
result_path = "./results/clustering_results/scanpy/"

os.makedirs(result_path, exist_ok=True)
def set_random_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    # If you are using other libraries that require a random seed, set them here as well

def run(dataset):
    try:
        print(dataset)
        set_random_seed(1)
        # adata for vis + clus
        # dataset_path = os.path.join(base_path, dataset, "run_1", 'test.h5ad')
        dataset_path = os.path.join(base_path, dataset, 'test.h5ad')

        adata = sc.read(dataset_path)
        print(adata)
        start = time.time()
        ## data is already normalized.
        if adata.X.max() > 20:
            sc.pp.normalize_total(adata, target_sum=1e4)
            sc.pp.log1p(adata)

        sc.pp.highly_variable_genes(adata, n_top_genes=2000)

        sc.tl.pca(adata)
        sc.pp.neighbors(adata)
        sc.tl.leiden(adata, flavor="igraph", n_iterations=2)
        # sc.tl.umap(adata)

        # calculate the silhouette score
        cluster_label = adata.obs['leiden']
        true_label = adata.obs['cell_type']

        # calculate the nmi, ari between true label and cluster label

        cluster_df = pd.DataFrame({'cell': adata.obs_names.to_list(), 'cluster': cluster_label, 'label': true_label})


        nmi = normalized_mutual_info_score(true_label, cluster_label)
        ari = adjusted_rand_score(true_label, cluster_label)
        # try:
        #     silhouette = silhouette_score(adata.obsm['X_umap'], true_label)
        #     silhouette_cluster = silhouette_score(adata.obsm['X_umap'], cluster_label)
        # except:
        silhouette = -1
        silhouette_cluster = -1
        ami = adjusted_mutual_info_score(true_label, cluster_label)

        # save it to a results.txt file
        path_to_save = os.path.join(result_path, dataset)
        if not os.path.exists(path_to_save):
            os.makedirs(path_to_save)
        
        ## save the cluster_df to a csv file
        cluster_df.to_csv(os.path.join(path_to_save, 'cluster_df.csv'), index=False)

        with open(path_to_save + '/results.txt', 'w') as f:
            f.write('Tissue: ' + dataset + '\n')
            f.write('NMI: ' + str(nmi) + '\n')
            f.write('ARI: ' + str(ari) + '\n')
            f.write('AMI: ' + str(ami) + '\n')
            f.write('silhouette: ' + str(silhouette) + '\n')
            f.write('silhouette_true_celltype: ' + str(silhouette) + '\n')
            f.write('silhouette_cluster: ' + str(silhouette_cluster) + '\n')
            f.write('Time: ' + str(time.time() - start) + '\n')
    except Exception as e:
        print(f"Error processing dataset {dataset}: {e}")
        with open(os.path.join(result_path, 'error_log.txt'), 'a') as f:
            f.write(f"Error processing dataset {dataset}: {e}\n")
        return None

if __name__ == '__main__':
    
    tissue_file = "/data/share/dungp/single-cell/ct-classification/valid_tissues.txt"
    with open(tissue_file, "r") as f:
        tissues = [line.strip() for line in f.readlines()]

    # tissues = os.listdir(base_path)
    remain_tissues = [t for t in tissues if not os.path.exists(os.path.join(result_path, t, "result.txt"))]
    # with mp.Pool(8) as p:
    #     results = p.map(run, remain_tissues)

    for tissue in remain_tissues:
        run(tissue)







