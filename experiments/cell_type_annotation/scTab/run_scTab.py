
#%%
import os

import argparse
# Set up argument parser
parser = argparse.ArgumentParser()
# parser.add_argument("--out_dir", type=str, default="scTab")
parser.add_argument("--device", type=int, default="cuda")
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--workers", type=int, default=4)
parser.add_argument("--cal_umap", action="store_true")
parser.add_argument('--data_dir', type=str, default='data', help='Directory for data')
parser.add_argument('--result_dir', type=str, default='models', help='Directory for saving results')

parser.add_argument("--parallel", '-mp', action="store_true", help="Run in parallel mode")

args = parser.parse_args()

# class ARGS:
#     def __init__(self):
#         self.out_dir = 'test'
#         self.gpu = 4
#         self.seed = 1
#         self.workers = 1
#         self.test = True
#         self.cal_silhouette = False
#         self.parallel = False
#         self.cal_umap = False

# args = ARGS()


os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ["BLAS_NUM_THREADS"] = "1"

os.environ["CUDA_VISIBLE_DEVICES"] = f"{args.device}"
from pathlib import Path
import time
import random

import numpy as np
import pandas as pd
import torch
import torchmetrics
from torch.utils.data import DataLoader

import pytorch_lightning as pl
import scanpy as sc
import sklearn
import multiprocessing as mp

from tabNet.tab_network import TabNet
from dataset import H5ADDataset, LOG1PTransform, TotalSumNormalize

def set_random_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



def determine_batch_size(num_samples):
    """
    Determine the batch size based on the number of samples in the dataset.
    
    Args:
        num_samples (int): Number of samples in the dataset.
    
    Returns:
        int: Batch size.
    """
    if num_samples > 100000:
        return 1024
    elif num_samples > 50000:
        return 512
    elif num_samples > 10000:
        return 256
    elif num_samples > 1000:
        return 128
    elif num_samples < 64:
        return 16
    else:
        return 64
   

def plot_umap(test_latent, test_labels):
    adata = sc.AnnData(test_latent)
    if len(test_labels) != test_latent.shape[0]:
        raise ValueError("Length of test_labels must match the number of rows in test_latent.")
    adata.obs['cell_type'] = pd.Categorical(test_labels)

    sc.pp.neighbors(adata, use_rep='X')
    sc.tl.umap(adata)
    

    # Adjust the legend to avoid it being cut off
    umap_plot = sc.pl.umap(
        adata, 
        color=['cell_type'], 
        frameon=False, 
        show=False, 
        return_fig=True
    )
    
    if umap_plot is not None:
        umap_plot.tight_layout(rect=[0, 0, 1, 1])  # Adjust layout to prevent cutoff

    return umap_plot, adata


# Set random seeds
set_random_seeds(args.seed)

# create lightining model which train classifier using tabnet as a backbone
class scTab(pl.LightningModule):
    def __init__(self, input_dim, n_classes, lr = 5e-3, weight_decay = 0.05, lambda_sparse = 1e-5, class_weight = None):
        super(scTab, self).__init__()
        self.save_hyperparameters()
        self.n_classes = n_classes
        self.lr = lr
        self.weight_decay = weight_decay
        self.lambda_sparse = lambda_sparse
        self.model = TabNet(input_dim=input_dim, output_dim=n_classes, 
            n_d=128,
            n_a=64,
            n_steps=1,
            gamma=1.3,
            n_independent=5,
            n_shared=3,
            virtual_batch_size=256,
            mask_type="entmax"
        )
        if class_weight is not None:
            self.criterion = torch.nn.CrossEntropyLoss(weight=torch.tensor(class_weight, dtype=torch.float32))
        else:
            self.criterion = torch.nn.CrossEntropyLoss()
        
        self.train_acc = torchmetrics.Accuracy(task='multiclass', num_classes=n_classes)
        
    
    def forward(self, x):
        return self.model(x)

    def extract_features(self, x):
        steps_output, M_loss = self.model.encoder(x)
        res = torch.sum(torch.stack(steps_output, dim=0), dim=0)
        return res

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat, m_loss = self.model(x)
        loss = self.criterion(y_hat, y) + self.lambda_sparse * m_loss
        acc = self.train_acc(y_hat, y)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_acc', acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        return optimizer
    


def predict(data_loader, encoder, label_mapping, device):
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
#%%
def run_tissue(dts_name, path_train, path_test, path_to_save, seed = 1):
    #%%
    # dts_name = 'adrenal tissue'
    print(f"Running dataset: {dts_name}")
    # Set path to save results
    checkpoint_path = path_to_save + "/checkpoints"
    os.makedirs(checkpoint_path, exist_ok=True)
    

    label_key = "cell_type_id"
    transform = torch.nn.Sequential(LOG1PTransform(), TotalSumNormalize(target_sum=1e4))


    train_ds = H5ADDataset(
        path_train, use_obs_column=label_key, transform=transform
    )

    train_label_dict = train_ds.label_dict
    train_int_to_label = {v: k for k, v in train_label_dict.items()}

    batch_size = determine_batch_size(len(train_ds))
    g = torch.Generator().manual_seed(seed)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True, worker_init_fn=lambda i: np.random.seed(seed+i), generator=g)
    
    
    if len(train_ds) < 50000:
        max_steps = 1000
    else:
        max_steps = 4000
    
    N_epochs = max_steps // (len(train_ds) // batch_size)


    # N_epochs = 100
    
    class_weight = None
    # Create model
    D_input = next(iter(train_loader))[0].shape[-1]
    n_classes = len(train_label_dict)
    if n_classes < 2:
        with open(os.path.join(path_to_save, "error.txt"), "a") as file:
            file.write(f"Tissue: {dts_name}\n")
            file.write("Not enough classes to train the model.")
        # return
    lr = 5e-3
    weight_decay = 0.05
    lambda_sparse = 1e-5
    wrapper = scTab(D_input, n_classes, lr, weight_decay, lambda_sparse, class_weight)
    # wrapper input: model, output_dim, num_classes, arc_s = 30, arc_m = 1, f_gamma = 2
    callbacks = [
        pl.callbacks.ModelCheckpoint(dirpath=checkpoint_path, monitor='train_loss', save_top_k=1, mode='min'),
        pl.callbacks.EarlyStopping(monitor='train_loss', patience=5, mode='min')
        ]
    trainer = pl.Trainer(
        max_epochs=N_epochs,
        accelerator='gpu',
        devices=[0],
        callbacks=callbacks,
        gradient_clip_algorithm='norm',
        gradient_clip_val=1.0,
    )

    start_time = time.time()
    # Train model
    trainer.fit(wrapper, train_loader)

    ## Evaluation
    test_ds = H5ADDataset(path_test, use_obs_column=label_key, transform=transform)

    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, drop_last=False, worker_init_fn=lambda i: np.random.seed(seed+i), generator=g)

    test_label_dict = test_ds.label_dict
    test_int_to_label = {v: k for k, v in test_label_dict.items()}

 
    # Predict on test data
    ## load the best checkpoint
    wrapper = scTab.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)
    wrapper.eval()
    wrapper.to('cuda:0')

    ## extract latents for plotting
    test_latents = []
    test_labels = []
    test_predictions = []
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to('cuda:0')
            test_latents.append(wrapper.extract_features(x).cpu().numpy())
            test_labels.append(y)
            test_predictions.append(wrapper(x)[0].cpu().numpy())
    test_latents = np.concatenate(test_latents)
    test_labels = np.concatenate(test_labels)
    test_labels = np.array([test_int_to_label[i] for i in test_labels])
    test_predictions = np.concatenate(test_predictions)
    ## convert logits to probabilities
    proba_predictions = torch.nn.functional.softmax(torch.tensor(test_predictions), dim=1).numpy()
    ## convert logits to integer labels
    test_predictions = np.argmax(proba_predictions, axis=1)
    test_predictions = np.array([train_int_to_label[i] for i in test_predictions])
    
    if args.cal_umap:
        try:
            fig, adata = plot_umap(test_latents, test_labels)
            # save umap plot
            fig.savefig(os.path.join(path_to_save, "umap.pdf"))
        except:
            adata = sc.AnnData(test_latents)
            if len(test_labels) != test_latents.shape[0]:
                raise ValueError("Length of test_labels must match the number of rows in test_latents.")
            adata.obs['cell_type'] = list(test_labels)
    else:
        adata = sc.AnnData(test_latents)
        if len(test_labels) != test_latents.shape[0]:
            raise ValueError("Length of test_labels must match the number of rows in test_latents.")
        adata.obs['cell_type'] = list(test_labels)

    unique_train_labels = set(train_label_dict.keys())
   

    idx2cal = [label in unique_train_labels for label in test_labels]
    idx2cal = np.array(idx2cal)

    if len(idx2cal) == 0:
        with open(os.path.join(path_to_save, "error.txt"), "w") as file:
            file.write(f"Tissue: {dts_name}\n")
            file.write("No common cell types between train and test data.")
        # return
    
    adata.obs['cell_type'] = list(test_labels)
    adata.obs['predictions'] = test_predictions
    adata.obs['idx2cal'] = idx2cal
    adata.obsm['proba_predictions'] = proba_predictions
    adata.write_h5ad(os.path.join(path_to_save, "pred.h5ad"))

    test_predictions, test_labels = test_predictions[idx2cal], np.array(test_labels)[idx2cal]

    acc = sklearn.metrics.accuracy_score(test_labels, test_predictions)
    f1 = sklearn.metrics.f1_score(test_labels, test_predictions, average="weighted")
    f1_macro = sklearn.metrics.f1_score(test_labels, test_predictions, average="macro")
    precision_weighted = sklearn.metrics.precision_score(test_labels, test_predictions, average="weighted")
    precision_macro = sklearn.metrics.precision_score(test_labels, test_predictions, average="macro")
    recall_weighted = sklearn.metrics.recall_score(test_labels, test_predictions, average="weighted")
    recall_macro = sklearn.metrics.recall_score(test_labels, test_predictions, average="macro")
    matthews_corrcoef_val = sklearn.metrics.matthews_corrcoef(test_labels, test_predictions)
    time_to_run = time.time() - start_time

    with open(os.path.join(path_to_save, "result.txt"), "w") as file:
        file.write(f"Tissue: {dts_name}\n")
        file.write(f"Number of cell types in train: {len(unique_train_labels)}\n")
        file.write(f"Number of cells in train: {len(train_ds)}\n")
        file.write(f"Number of cell types in test: {np.unique(test_labels).shape[0]}\n")
        file.write(f"Number of cells in test: {len(test_ds)}\n")
        file.write(f"Number of cell types in common: {len(unique_train_labels.intersection(set(test_labels)))}\n")
        # train and test metrics

        file.write(f"Accuracy: {acc}\n")
        file.write(f"F1 Weighted: {f1}\n")
        file.write(f"F1 Macro: {f1_macro}\n")
        file.write(f"Precision (weighted): {precision_weighted}\n")
        file.write(f"Precision Macro: {precision_macro}\n")
        file.write(f"Recall (weighted): {recall_weighted}\n")
        file.write(f"Recall Macro: {recall_macro}\n")
        file.write(f"Matthews Correlation Coefficient: {matthews_corrcoef_val}\n")
        file.write(f"Total Time: {time_to_run}\n")
        # Wrapper model param
        file.write(f"lr: {lr}\n")
        file.write(f"weight_decay: {weight_decay}\n")
        file.write(f"Device: {args.device}\n")
        file.write(f"Seed: {seed}\n")
        file.write(f"Workers: {args.workers}\n")
        file.write(f"N_epochs: {N_epochs}\n")

    torch.cuda.empty_cache()
    print("Script ended")

    # except Exception as e:
    #     # log the failure and error to a file
    #     error_file = os.path.join(failed_path, f"{tissue}.txt")
    #     with open(error_file, "w") as file:
    #         file.write(str(e))
    #     return

def get_path_to_save(save_root, dts_name):
    # Define the path to save results
    path_to_save = os.path.join(save_root, dts_name)
    # Create the directory if it does not exist
    return path_to_save

def job_fn(args):
    tissue, path_train, path_test, path_to_save, seed = args
    try:
        set_random_seeds(seed)

        pl.seed_everything(seed, workers=True)
        run_tissue(tissue, path_train, path_test, path_to_save, seed)
    except Exception as e:
        print(f"Failed {tissue}: {e}")
#%%

if __name__ == "__main__":
    mp.set_start_method('spawn')
    DATA_DIR = Path(args.data_dir)
    RESULTS_DIR = Path(args.result_dir)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    

    data_dir = DATA_DIR / 'cellxgene'
    tissues = os.listdir(data_dir)
    tissues = tissues[0:1]
    save_root = RESULTS_DIR / "main_results" / "scTab" 
    save_root.mkdir(parents=True, exist_ok=True)

    path_train_template = "{tissue}/train.h5ad"
    path_test_template = "{tissue}/test.h5ad"

    # Filter remaining tissues
    remain_tissues = []
    for tissue in tissues:
        result_file = os.path.join(get_path_to_save(save_root, tissue), "result.txt")
        if os.path.exists(result_file):
            continue
        remain_tissues.append(tissue)

    print(f"Number of tissues to run: {len(remain_tissues)}")

    # Construct job list
    jobs = []
    for tissue in remain_tissues:
        path_train = data_dir / path_train_template.format(tissue=tissue)
        path_test = data_dir / path_test_template.format(tissue=tissue)
        path_to_save = get_path_to_save(save_root, tissue)

        if not os.path.exists(path_train) or not os.path.exists(path_test):
            print(f"Data files for {tissue} do not exist. Skipping...")
            continue

        jobs.append((tissue, path_train, path_test, path_to_save, args.seed))

    start_time = time.time()

    if args.parallel:
        print("Running in parallel mode")
        with mp.Pool(args.workers) as pool:
            pool.map(job_fn, jobs)
    else:
        print("Running in sequential mode")
        os.makedirs(os.path.join(save_root, "failed"), exist_ok=True)
        for job in jobs:
            tissue = job[0]
            try:
                print(f"Running tissue: {tissue}")
                job_fn(job)
            except Exception as e:
                print(f"Error running tissue {tissue}: {e}")
                with open(os.path.join(save_root, "failed", f"{tissue}.txt"), "w") as f:
                    f.write(str(e))

    total_time = time.time() - start_time
    print(f"Total time: {total_time:.2f} seconds")

    # Log args + time
    os.makedirs(save_root, exist_ok=True)
    with open(os.path.join(save_root, "args.txt"), "w") as f:
        f.write(str(args))
        f.write(f"\nTotal time: {total_time:.2f} seconds")

    print("Script ended.")

# %%
