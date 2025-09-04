
#%%
import os
# Environment settings
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['BLAS_NUM_THREADS'] = '1'

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import scanpy as sc
from sclmct.utils import cluster_and_plot_umap
import torch
import numpy as np
import scanpy as sc

from tqdm import tqdm



def get_latent_representations(data_loader, encoder, label_mapping, device):
    encoder = encoder.to(device)
    encoder.eval()
    latent_reps, labels = [], []
    with torch.no_grad():
        for x, y in data_loader:
            x = x.to(device)
            z = encoder(x)
            latent_reps.append(z.cpu().numpy())
            labels.extend([label_mapping[i].replace("_", " ").lower() for i in y.numpy()])
    return np.concatenate(latent_reps), np.array(labels)


data_root = './data/cellxgene'
result_dir = "./results/main_results/scLMCT"

batch_size = 512

save_fold = "/data/share/dungp/single-cell/ct-classification/code4publication/code-for-reproducibility/test/clustering_results"

#%%

#%%
with open("./data/valid_tissues.txt", 'r') as f:
    tissues = [line.strip() for line in f.readlines()]

for dts_name in tqdm(tissues):

    # try:
    ## just for testing
    save_path = save_fold + f"/{dts_name}"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    save_file = f"{save_path}/clustering_result.txt"
    dts_fold = "./results/main_results/scLMCT" + f"/{dts_name}"
    if os.path.exists(save_file):
        continue

    pred_path = f'{dts_fold}/pred.h5ad'


    adata = sc.read_h5ad(pred_path)
    ## extract embeddings of training and testing datasets
    latent_test = adata.X
    test_labels = adata.obs['true_labels'].values

    nmi, ari, silhouette, silhouette_cluster, ami, exec_time, clst_adata = cluster_and_plot_umap(latent_test, test_labels, cal_umap = False, cal_silhouette=False, merge_threshold=0.8)

    adata.obs['leiden'] = clst_adata.obs['leiden']
    adata.obs['merged_leiden'] = clst_adata.obs['merged_leiden']
    
    # sc.pp.neighbors(adata, use_rep='X')
    # sc.tl.umap(adata)
    # sc.pl.umap(adata, color = ['true_labels', 'leiden', 'merged_leiden'])

    ## overwrite the adata with clustering results
    adata.write_h5ad(pred_path)

    
    # save_path = f"{save_root}/analysis_1/{dts_name}"
    # if not os.path.exists(save_path):
    #     os.makedirs(save_path)
    with open(save_file, 'w') as f:
        f.write(f"NMI: {nmi}\n")
        f.write(f"ARI: {ari}\n")
        f.write(f"AMI: {ami}\n")
        f.write(f"Execution Time: {exec_time}\n")
    # except:
    #     continue
# %%
