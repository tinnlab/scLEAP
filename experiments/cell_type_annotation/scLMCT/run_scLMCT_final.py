#%%
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ["BLAS_NUM_THREADS"] = "1"

# ========== ARGUMENT CLASS ==========
class ARGS:
    def __init__(self, l_lambda, use_proj, block_type, arc_m, arc_s, text_encoder_model, data_dir):
        self.out_dir = "Ablation"
        self.classifier = "faiss"
        self.workers = 12
        self.seed = 1
        self.fold = 0
        self.device = 4
        self.max_epoch = 200
        self.cal_umap = False
        self.cal_silhouette = False
        self.parallel = False
        self.test = False
        self.test_tissue = 'temporal cortex'
        self.loss_type = 'combine'
        self.l_lambda = l_lambda
        self.use_projection_head = use_proj
        self.block_type = block_type
        self.arc_m = arc_m
        self.arc_s = arc_s
        self.clip_loss_type = 'label'
        self.prompt_template = "cell_types_info"
        self.text_encoder_model = text_encoder_model
        self.save_root = "/nfs/blanche/dungp/single-cell/cell_annotation/results"
        self.data_dir = data_dir

# # ========== ARGUMENT PARSER ==========
# def get_arg_parser():
#     parser = argparse.ArgumentParser(description="scLMCT Final Run")
#     parser.add_argument('--l_lambda', type=float, default=0.5, help='Lambda for loss combination')
#     parser.add_argument('--use_proj', action='store_true', help='Use projection head')
#     parser.add_argument('--block_type', type=str, default='v2', help='Block type')
#     parser.add_argument('--arc_m', type=float, default=0.4, help='ArcFace margin')
#     parser.add_argument('--arc_s', type=float, default=30.0, help='ArcFace scale')
#     parser.add_argument('--text_encoder_model', type=str, default='michiyasunaga/BioLinkBERT-base', help='Text encoder model')
#     parser.add_argument('--data_dir', type=str, required=True, help='Path to data directory')
#     parser.add_argument('--save_root', type=str, default="/nfs/blanche/dungp/single-cell/cell_annotation/results", help='Root directory to save results')
#     parser.add_argument('--workers', type=int, default=12, help='Number of workers for parallel processing')
#     parser.add_argument('--seed', type=int, default=1, help='Random seed')
#     parser.add_argument('--max_epoch', type=int, default=200, help='Maximum number of epochs')
#     parser.add_argument('--parallel', action='store_true', help='Run tissues in parallel')
#     parser.add_argument('--device', type=int, default=4, help='CUDA device ID')
#     parser.add_argument('--prompt_template', type=str, default="cell_types_info_v1", help='Prompt template name')
#     parser.add_argument('--loss_type', type=str, default='combine', help='Loss type')
#     parser.add_argument('--clip_loss_type', type=str, default='label', help='CLIP loss type')
#     return parser

# parser = get_arg_parser()
# args = parser.parse_args()
# os.environ['CUDA_VISIBLE_DEVICES'] = f'{args.device}'  # Set to your desired GPU ID
os.environ['CUDA_VISIBLE_DEVICES'] = '3'  # Set to your desired GPU ID


import pytorch_lightning as pl
pl.seed_everything(1, workers=True)  # Set a fixed seed for reproducibility
import random
import numpy as np
import pandas as pd
import torch
import argparse
import warnings
import json
import time
import scanpy as sc 
import multiprocessing as mp
import gc
from itertools import product
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, classification_report
import faiss

from sclmct.model import TrainWrapperCLIPStyle
from sclmct.dataset import H5ADDataset, LOG1PTransform, TotalSumNormalize
from sclmct.utils import predict_with_distance_weights
from sclmct.paths import DATA_DIR, MODELS_DIR, RESULTS_DIR, CONFIGS_DIR, ensure_dirs


def determine_batch_size(n):
    if n > 100_000: return 4096
    if n > 50_000: return 512
    if n > 10_000: return 256
    if n > 1_000: return 128
    if n < 64: return 16
    return 64


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


def compute_metrics(y_true, y_pred):
    print(classification_report(y_true, y_pred, zero_division=0))
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "F1 Weighted": f1_score(y_true, y_pred, average="weighted"),
        "F1 Macro": f1_score(y_true, y_pred, average="macro")
    }


def get_semantics_labels(prompt_template, train_label_dict, ols_mappings=None):
     ## return a default set of prompts
    simple_templates = [
        # SHORT TEMPLATES (20)
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
        prompts_path = f"./scLMCT/ct_descriptions/{prompt_template}.json"
        label_prompts = json.load(open(prompts_path, 'r'))
        label_prompts = {k: v for k, v in label_prompts.items() if ols_mappings[k] in train_label_dict}
        assert len(label_prompts) < len(train_label_dict), f"There're cell types that're not in {prompt_template}.json"
        label_prompts = {train_label_dict[ols_mappings[k]]: v for k, v in label_prompts.items() if k in ols_mappings}

       
        ## if 'unknown' label is present, add specific prompts
        if 'unknown' in train_label_dict:
            semantics_labels[train_label_dict['unknown']] = [
                "This unknown cell population does not match any known cell type and may represent a novel or intermediate state.",
                "This unknown cell population expresses mixed lineage markers and may represent a novel or intermediate state.",
                "This unknown cell population has ambiguous classification with low confidence, suggesting a novel or intermediate state.",
                "This unknown cell population is uncharacterized and may represent a rare or intermediate state.",
                "This unknown cell population lacks canonical markers and may represent a novel or intermediate state.",
                "This unknown cell population is not captured in current ontologies and may represent a novel or intermediate state.",
                "This unknown cell population partially resembles immune subtypes but remains unclassified, suggesting a novel or intermediate state.",
                "This unknown cell population forms a distinct cluster and may represent a novel or intermediate state.",
                "This unknown cell population lacks unique markers and may represent a heterogeneous or intermediate state.",
                "This unknown cell population may reflect stress, doublets, or noise, and is best described as a novel or intermediate state."
            ]

        #  ## if any label is missing, fill it with simple templates
        # if len(label_prompts) < len(train_label_dict):
        #     missing_labels = set(train_label_dict.values()) - set(semantics_labels.keys())
        #     for missing_label in missing_labels:
        #         semantics_labels[missing_label] = [
        #             template.format(label=missing_label) for template in simple_templates
        #         ]
        ## sort the keys to ensure consistent order, important during training
        semantics_labels = {k: label_prompts[k] for k in sorted(label_prompts.keys())}

    except:
        semantics_labels = {}
        for label, id in train_label_dict.items():
            label_prompts = [template.format(label=label) for template in simple_templates]
            semantics_labels[id] = label_prompts
    return semantics_labels

def create_faiss_index(train_lat, use_gpu=True, gpu_id=0):
    d = train_lat.shape[1]
    index_cpu = faiss.IndexFlatIP(d)  # Inner product

    if use_gpu:
        res = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(res, gpu_id, index_cpu)
    else:
        index = index_cpu

    index.add(train_lat.astype(np.float32))
    return index


def get_path_to_save(args, tissue):
    base_path = f"{args.save_root}"

    
    save_path = os.path.join(base_path, tissue)
        
    return save_path, base_path

#%%
def run_tissue(tissue, args):
    #%%
    print(f"Running tissue: {tissue}")
    torch.set_float32_matmul_precision("high")
    torch.use_deterministic_algorithms(True, warn_only=True)

    PATH_TO_SAVE, _ = get_path_to_save(args, tissue)
    os.makedirs(PATH_TO_SAVE, exist_ok=True)
    
    PATH_TRAIN = f"{args.data_dir}/{tissue}/train.h5ad"
    PATH_TEST = f"{args.data_dir}/{tissue}/test.h5ad"
    

    
    ols_mappings = json.load(open(DATA_DIR / "ct_mappings" / "improved_cellname2id_mappings.json", 'r'))
    ols_mappings = {k.lower(): v.lower() for k, v in ols_mappings.items()}
    inverse_ols_mappings = {v.lower(): k.lower() for k, v in ols_mappings.items()}
    transform = torch.nn.Sequential(LOG1PTransform(), TotalSumNormalize(target_sum=1e4))

    train_ds = H5ADDataset(PATH_TRAIN, use_obs_column="cell_type", ols_mappings=ols_mappings, transform=transform)
    train_label_dict = train_ds.label_dict

    if len(train_label_dict) < 2:
        with open(os.path.join(PATH_TO_SAVE, "error.txt"), "w") as f:
            f.write("Not enough classes for training.")
        # return
    

    semantics_labels = get_semantics_labels(args.prompt_template, train_label_dict, ols_mappings)
    batch_size = determine_batch_size(len(train_ds))
    g = torch.Generator().manual_seed(args.seed)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True,
                              worker_init_fn=lambda i: np.random.seed(args.seed+i), generator=g)

    train_ds_eval = H5ADDataset(PATH_TRAIN, use_obs_column="cell_type", transform=transform)
    test_ds = H5ADDataset(PATH_TEST, use_obs_column="cell_type", transform=transform)

    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, drop_last=False,
                             worker_init_fn=lambda i: np.random.seed(args.seed+i), generator=g)

    cache_path = os.path.join(PATH_TO_SAVE, "cached_prompt_embeddings.pt")
    D_input = next(iter(train_loader))[0].shape[-1]

    ## check if the checkpoint folder is not empty
    if os.path.exists(os.path.join(PATH_TO_SAVE, "checkpoints")) and len(os.listdir(os.path.join(PATH_TO_SAVE, "checkpoints"))) > 0:
        print("Checkpoints already exist, skipping training.")
        ## get the checkpoint path (usually only one checkpoint is saved)
        checkpoint_path = os.path.join(PATH_TO_SAVE, "checkpoints", os.listdir(os.path.join(PATH_TO_SAVE, "checkpoints"))[0])
        model = TrainWrapperCLIPStyle.load_from_checkpoint(checkpoint_path)
    else:
        print("Training new model.")
        ## create a new model
        model = TrainWrapperCLIPStyle(
            input_dim=D_input,
            hidden_dim=512,
            output_dim=None,
            label_prompts=semantics_labels,
            n_cell_layers=2,
            block_type=args.block_type,
            text_encoder_model=args.text_encoder_model,
            use_projection_head=args.use_projection_head,
            n_prompts=5,
            pool_type="mean",
            n_text_layers_to_finetune=1,
            device="cuda",
            cache_path=cache_path,
            lr=1e-4,
            weight_decay=1e-4,
            arc_s=args.arc_s,
            arc_m=args.arc_m,
            f_gamma=1.3,
            loss_type=args.loss_type,
            l_lambda=args.l_lambda,
            clip_loss_type=args.clip_loss_type,
        )

        callbacks = [
            pl.callbacks.EarlyStopping(monitor="train_loss", patience=5, min_delta=0.001),
            pl.callbacks.ModelCheckpoint(dirpath=os.path.join(PATH_TO_SAVE, "checkpoints"),
                                        monitor="train_loss", save_top_k=1, mode="min")
        ]

        trainer = pl.Trainer(
            max_epochs=args.max_epoch,
            accelerator="gpu",
            devices=[0],
            callbacks=callbacks,
            gradient_clip_val=1.0,
            deterministic=True,
        )

        start = time.time()
        trainer.fit(model, train_loader)
        model = TrainWrapperCLIPStyle.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)
        print(f"Training finished in {time.time() - start:.2f} seconds.")

    start_latent_infer = time.time()
    print("Getting latent representations for training and testing datasets...")
    train_lat, train_lbls = get_latent_representations(
        DataLoader(train_ds_eval, batch_size=batch_size, shuffle=False, drop_last=False, generator=g),
        model,
        train_ds_eval.get_index_to_label_mapping(),
        "cuda"
    )

    test_lat, test_lbls = get_latent_representations(
        test_loader,
        model,
        test_ds.get_index_to_label_mapping(),
        "cuda"
    )
    print(f"Latent representations obtained in {time.time() - start_latent_infer:.2f} seconds.")


    
    start_text_embed = time.time()
    print('Calculating semantic embeddings...')
    model.setup()
    with torch.no_grad():
        sem_lbls, sem_embeds = model.get_class_text_centers(only_centers=True)
    print(f"Semantic embeddings calculated in {time.time() - start_text_embed:.2f} seconds.")

    sem_lbls = sem_lbls.cpu().numpy()
    sem_embeds = sem_embeds.cpu().numpy()
    faiss.normalize_L2(sem_embeds)

    sem_lbls = [train_ds.get_index_to_label_mapping()[i] for i in sem_lbls]
    sem_lbls = [lb.lower().replace("_", " ") for lb in sem_lbls]

    train_lbls_id = np.array([ols_mappings[x] for x in train_lbls])

    start_building_index = time.time()
    print("Building FAISS indices for semantic and latent embeddings...")
    # latent_index = IndexFlatIP(train_lat.shape[1])
    # latent_index.add(train_lat.astype(np.float32))
    # print(f"FAISS index for latent embeddings built in {time.time() - start_building_index:.2f} seconds.")
    # semantic_index = IndexFlatIP(sem_embeds.shape[1])
    # semantic_index.add(sem_embeds.astype(np.float32))
    latent_index = create_faiss_index(train_lat.astype(np.float32), use_gpu=True)
    print(f"FAISS index for latent embeddings built in {time.time() - start_building_index:.2f} seconds.")
    semantic_index = create_faiss_index(sem_embeds.astype(np.float32), use_gpu=True)

    print(f"FAISS index for semantic embeddings built in {time.time() - start_building_index:.2f} seconds.")

    print("Predicting with FAISS indices...")
    start_predicting = time.time()
    D_sem, I_sem = semantic_index.search(test_lat.astype(np.float32), 1)
    pred_sem = [sem_lbls[i[0]] for i in I_sem]

    pred_lat, D_lat = predict_with_distance_weights(
        latent_index, test_lat.astype(np.float32), train_lbls_id, 11, return_distances=True
    )
    print(f"Predictions made in {time.time() - start_predicting:.2f} seconds.")

    ## plot the umap landscape of test cell embeddings and text embeddings
    # sem_lbls_name = [inverse_ols_mappings[x] for x in sem_lbls]
    # plot_embeddings(test_lat, test_lbls, sem_embeds, sem_lbls_name, save_path=os.path.join(PATH_TO_SAVE, "test_embeddings_umap.png"))

    print("Creating prediction AnnData object...")
    start_andata_pred = time.time()
    # pred_adata = sc.AnnData(test_lat)
    # pred_adata.obs['true_labels'] = test_lbls
    # pred_adata.obs['true_labels_id'] = [ols_mappings[x] for x in test_lbls]
    # pred_adata.obs['pred_labels_faiss_sem'] = [inverse_ols_mappings[x] for x in pred_sem]
    # pred_adata.obs['pred_labels_faiss_sem_id'] = pred_sem
    # pred_adata.obs['pred_labels_faiss_lat'] = [inverse_ols_mappings[x] for x in pred_lat]
    # pred_adata.obs['pred_labels_faiss_lat_id'] = pred_lat
    # pred_adata.obsm['faiss_similarity_sem'] = D_sem
    # pred_adata.obsm['faiss_similarity_lat'] = D_lat


    # Build obs DataFrame in one go
    obs_df = pd.DataFrame({
        'true_labels': test_lbls,
        'true_labels_id': [ols_mappings[x] for x in test_lbls],
        'pred_labels_faiss_sem_id': pred_sem,
        'pred_labels_faiss_lat_id': pred_lat,
    })

    # Vectorized reverse mapping
    obs_df['pred_labels_faiss_sem'] = obs_df['pred_labels_faiss_sem_id'].map(inverse_ols_mappings)
    obs_df['pred_labels_faiss_lat'] = obs_df['pred_labels_faiss_lat_id'].map(inverse_ols_mappings)

    
    obs_df['idx2cal'] = [x in set(train_lbls_id) for x in [ols_mappings[y] for y in test_lbls]]
    idx2cal = np.where(obs_df['idx2cal'])[0]

    # Create AnnData efficiently
    pred_adata = sc.AnnData(X=test_lat, obs=obs_df)
    end_andata_pred = time.time()
    print(f"AnnData creation took {end_andata_pred - start_andata_pred:.2f} seconds")
    pred_adata.write_h5ad(os.path.join(PATH_TO_SAVE, "pred.h5ad"))

    # # Save latent matrix (with compression)
    # np.savez_compressed(os.path.join(PATH_TO_SAVE, "test_lat.npz"), test_lat=test_lat)
    # # Save obs DataFrame
    # obs_df.to_parquet(os.path.join(PATH_TO_SAVE, "obs_df.parquet"), index=False)
    print("Calculating metrics...")
    start_metrics = time.time()
    y_true = np.array(obs_df['true_labels_id'])[idx2cal]
    y_sem = np.array(obs_df['pred_labels_faiss_sem_id'])[idx2cal]
    y_lat = np.array(obs_df['pred_labels_faiss_lat_id'])[idx2cal]

    metrics_sem = compute_metrics(y_true, y_sem)
    metrics_lat = compute_metrics(y_true, y_lat)

    with open(os.path.join(PATH_TO_SAVE, "result.txt"), "w") as f:
        f.write(f"Tissue: {tissue}\n")
        f.write("\n=== FAISS on Semantic Embeddings ===\n")
        for k, v in metrics_sem.items():
            f.write(f"{k}: {v:.4f}\n")
        f.write("\n=== FAISS on Latent Embeddings ===\n")
        for k, v in metrics_lat.items():

            f.write(f"{k}: {v:.4f}\n")
        f.write(f"\nTime: {time.time() - start:.2f} sec\n")
        
    print(f"Metrics calculated in {time.time() - start_metrics:.2f} seconds.")
    

    del model
    torch.cuda.empty_cache()
    gc.collect()
    #%%

#%%
def job_fn(args):
    tissue, args = args
    import random, numpy as np, torch, pytorch_lightning as pl
    pl.seed_everything(args.seed, workers=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    try:
        run_tissue(tissue, args)
    except Exception as e:
        print(f"Failed {tissue}: {e}")
#%%
# ========== MAIN FUNCTION ==========
def run_all():
    mp.set_start_method('spawn')
    warnings.filterwarnings("ignore")

    # tissue = 'nasopharynx'
    config = {
        "l_lambda": 0.5,
        "use_proj": False,
        "block_type": 'v2',
        "arc_m": 0.4,
        "arc_s": 30.0,
    }

    
    data_dir = DATA_DIR / 'cellxgene'
    tissues = os.listdir(data_dir)
    tissues = tissues[:2]

    text_encoder_model = 'michiyasunaga/BioLinkBERT-base'
    l_lambda, use_proj, block_type, arc_m, arc_s = config['l_lambda'], config['use_proj'], config['block_type'], config['arc_m'], config['arc_s']

    args = ARGS(l_lambda, use_proj, block_type, arc_m, arc_s, text_encoder_model, data_dir)

    args.save_root = RESULTS_DIR / "main_results" / "scLMCT"
    args.save_root.mkdir(parents=True, exist_ok=True)
    
    _, path_base = get_path_to_save(args, tissues[0])
    remain_tissues = []

    for tissue in tissues:
        result_file = os.path.join(path_base, tissue, "result.txt")
        if not os.path.exists(result_file):
            remain_tissues.append(tissue)
    
    print("Total of {} tissues to run.".format(len(remain_tissues)))
    jobs = [(tissue, args) for tissue in remain_tissues]
    
    pl.seed_everything(args.seed, workers=True)
            
    start_time = time.time()

    if args.parallel:
        print(f"Running in parallel with {args.workers} workers")
        with mp.Pool(args.workers) as pool:
            pool.map(job_fn, jobs)
            pool.close()
            pool.join()
            del pool
            gc.collect()
    else:
        for job in jobs:
            job_fn(job)

    total_time = time.time() - start_time
    with open(f"{path_base}/args.txt", "w") as file:
        file.write(str(vars(args)))
        file.write(f"\nTotal time: {total_time:.2f} seconds")
    print(f"Finished: {path_base}\nTotal time: {total_time:.2f} seconds")

            

#%%
if __name__ == "__main__":
    run_all()
    