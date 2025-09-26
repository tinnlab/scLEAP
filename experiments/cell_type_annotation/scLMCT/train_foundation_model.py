

#%%
import sys 
sys.path.append('/data/dungp/projects/single-Cell/Sy-Code/src_LLM_CLIP')
import os

# Environment settings
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['BLAS_NUM_THREADS'] = '1'

import argparse
parser = argparse.ArgumentParser(description="Run training and evaluation for a specific seed.")
parser.add_argument("-s", "--seed", type=int, required=True, help="Seed value for the run.")
parser.add_argument("-r", "--run_id", type=str, required=True, help="Unique identifier for the run.")
parser.add_argument("-l", "--loss_func", type=str, default='arcface', help="Which loss function to use. arcface or curricular")
parser.add_argument("-g", "--gpu", type=int, required=True, help="GPU device to use.")
parser.add_argument("-e", "--max_epochs", type=int, default=20, help="Maximum number of epochs for training.")
parser.add_argument("-d", "--data_fold", type=str, required=True, help="Folder of the parquets, e.g. 'foundation-training-data'")

args = parser.parse_args()

# class ARGS:
#     def __init__(self):
#         self.seed = 1
#         self.run_id = 'test'
#         self.loss_func = 'combine'
#         self.gpu = '1'
#         self.max_epochs = 20
#         self.data_fold = "data-processed-by-Dung-min-count-100-balanced-max5000"
# args = ARGS()

os.environ['CUDA_VISIBLE_DEVICES'] = f'{args.gpu}'
from numba import cuda
 
cuda.select_device(0)
import gc
import json
import time
import datetime
import random
import numpy as np
import torch
import os
from math import ceil
from os.path import join
from typing import Dict, List
from tqdm import tqdm
import merlin.io
from merlin.loader.torch import Loader
from merlin.dtypes import boolean
from merlin.dtypes import float32, int64
from merlin.schema import ColumnSchema, Schema



import pytorch_lightning as pl
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
from pytorch_lightning.loggers import CSVLogger
from sclmct.model import TrainWrapperCLIPStyle

import cupy

#%%
parquet_root = "./data/"


PARQUET_SCHEMA = {
    'X': float32,
    # 'cell_type': int64,
    'ontology_int': int64,

}


# -------------------------------
# 1. Configuration and Settings
# -------------------------------
ols_json_path = "./data/ct_mappings/cellname2id.json"
ols_mappings = json.load(open(ols_json_path, 'r'))
ols_to_int = json.load(open(f"./foundation_model/cell_type_to_int.json", 'r'))

## path to the cell type description json file
prompts_path = "./foundation_model/cell_type_descriptions.json"
label_prompts = json.load(open(prompts_path, 'r'))
label_prompts = {ols_to_int[ols_mappings[k]]: v for k, v in label_prompts.items() if k in ols_mappings and ols_mappings[k] in ols_to_int}


## sort the keys of the label prompts
label_prompts = {k: label_prompts[k] for k in sorted(label_prompts.keys())}


## align the keys of the label prompts with the 
# Model configuration settings
settings = {
    "input_dim": 11852, ## number of input genes
    "hidden_dim": 512, 
    "output_dim": 768,
    "label_prompts": label_prompts,  
    "n_prompts": 5,
    "train_layers": [10, 11], # Layers to fine-tune, e.g., [10, 11] for the last two layers
    'pool_type': 'mean',  # Choose between 'mean' and 'cls'
    "lr": 1e-4,
    "weight_decay": 1e-5,
    "arc_s": 30,
    "arc_m": 0.4,
    "f_gamma": 1, ## focal loss gamma
    "loss_type": f"{args.loss_func}",  # Choose between "arcface", "curricular", or "combine"
    "device": 'cuda:0',
    "batch_size": 4096 * 2,
    "epochs": args.max_epochs,
    "class_weights": None,  # Replace with actual class weights if needed
    "l_lambda": 0.5,  # Weight for text loss
    "cache_path": "./cache/cached_prompt_embeddings.pt",  # Path to cache directory
    "clip_loss_type": "label"  # Type of CLIP loss to use
}





def merlin_dataset_factory(path: str, columns: List[str], dataset_kwargs: Dict[str, any]):
    return merlin.io.Dataset(
        path,
        engine='parquet',
        schema=Schema(
            [
                ColumnSchema(
                    'X', dtype=PARQUET_SCHEMA['X'],
                    is_list=True, is_ragged=False,
                    properties={'value_count': {'max': 11852}}
                )
            ] +
            [ColumnSchema(col, dtype=PARQUET_SCHEMA[col]) for col in columns]
        ),
        **dataset_kwargs
    )

def set_default_kwargs_dataloader(kwargs: Dict[str, any] = None, training: bool = True):
    assert isinstance(training, bool)
    if kwargs is None:
        kwargs = {}
    if 'parts_per_chunk' not in kwargs:
        kwargs['parts_per_chunk'] = 4 if training else 1
    if 'drop_last' not in kwargs:
        kwargs['drop_last'] = training
    if'shuffle' not in kwargs:
        kwargs['shuffle'] = training

    return kwargs


def set_default_kwargs_dataset(kwargs: Dict[str, any] = None, training: bool = True):
    if kwargs is None:
        kwargs = {}
    if all(['part_size' not in kwargs, 'part_mem_fraction' not in kwargs]):
        kwargs['part_size'] = '1GB' if training else '1GB'

    return kwargs

def get_loaders(seed, train_parquets, batch_size):
    dataset_kwargs_train = set_default_kwargs_dataset(training=True)
    loader_kwargs_train = set_default_kwargs_dataloader(training=True)

    # Define dataset columns
    # columns = ["cell_type"]
    columns = ["ontology_int"]

    
    train_dataset = merlin_dataset_factory(train_parquets, columns, dataset_kwargs_train)

    train_loader = merlin.loader.torch.Loader(
        train_dataset, batch_size=batch_size, seed_fn=seed_worker(seed), **loader_kwargs_train
    )

    return train_loader



# Usage example


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)



#%%
def run_seed(run_id, seed):
    run_id = args.run_id
    seed = args.seed
    # Set the seed for reproducibility and initialize logger
    seed_worker(seed)
    run_root = "./scLMCT_foundation"
    run_root = os.path.join(run_root, run_id)
    os.makedirs(run_root, exist_ok=True)
    with open(os.path.join(run_root, "settings.json"), "w") as f:
        json.dump(settings, f, indent=4)

    if settings['class_weights']:
        settings['class_weights'] = torch.tensor(settings['class_weights']).float()
    # Create folders for logs, checkpoints, embeddings, faiss indexes, and evaluation metrics
    folders = ['logs', 'checkpoints']
    for folder in folders:
        os.makedirs(os.path.join(run_root, folder), exist_ok=True)
    log_fold, checkpoint_fold = [os.path.join(run_root, folder) for folder in folders]

    logger = CSVLogger(log_fold, version=f"run_{seed}", name="mamba_training")

    train_parquets = [os.path.join(parquet_root, f'{args.data_fold}', file) for file in os.listdir(os.path.join(parquet_root, f'{args.data_fold}')) if file.endswith('.parquet')]
    train_loader = get_loaders(seed, train_parquets, settings['batch_size'])

    
    # remove the settings['cache_path'] file
    if os.path.exists(settings['cache_path']):
        os.remove(settings['cache_path'])

    model = TrainWrapperCLIPStyle(
        # encoder = None,
        input_dim=settings['input_dim'],
        hidden_dim=settings['hidden_dim'],
        output_dim=settings['output_dim'],
        label_prompts=settings['label_prompts'],
        n_prompts=settings['n_prompts'],
        pool_type=settings['pool_type'],
        lr=settings['lr'],
        weight_decay=settings['weight_decay'],
        arc_s=settings['arc_s'],
        arc_m=settings['arc_m'],
        f_gamma=settings['f_gamma'],
        l_lambda=settings['l_lambda'],
        device=settings['device'],
        cache_path=settings['cache_path'],
        n_cell_layers=2,  ## number of MLP residual blocks in the cell encoder
        block_type='v2',
        text_encoder_model="cambridgeltl/SapBERT-from-PubMedBERT-fulltext",
        use_projection_head=False,
        n_text_layers_to_finetune=4
    ) 
    
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=os.path.join(checkpoint_fold, f"run_{seed}"),
        monitor="train_loss", mode="min", save_top_k=5, save_weights_only=False
    )

    trainer = pl.Trainer( 
        max_epochs=settings["epochs"], 
        accelerator='gpu', 
        devices=[0], 
        logger=logger, 
        # callbacks=[early_stop_callback, checkpoint_callback], 
        callbacks=[checkpoint_callback],
        accumulate_grad_batches=1, 
        gradient_clip_val=1.0, 
        log_every_n_steps=200,
        precision="16-mixed",
        deterministic=True, 
        benchmark=False,
        num_sanity_val_steps=0
        # strategy="ddp"

    )

    trainer.fit(model, train_loader)

  


if __name__ == "__main__":


    run_seed(args.run_id, args.seed)
    



















