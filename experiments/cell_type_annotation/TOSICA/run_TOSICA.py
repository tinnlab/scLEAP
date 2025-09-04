#%%
import os
from pathlib import Path
import sys
import argparse
import random
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from sklearn.preprocessing import LabelEncoder
from collections import OrderedDict
import scanpy as sc
import anndata as ad
import time
import math
from torch.utils.data import Dataset
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.tensorboard import SummaryWriter
from TOSICA.TOSICA_model import scTrans_model as create_model

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

parser = argparse.ArgumentParser()
parser.add_argument('--device', type=str, default='1', help='GPU ID to use')
parser.add_argument('--data_dir', type=str, default='data', help='Directory for data')
parser.add_argument('--result_dir', type=str, default='models', help='Directory for saving results')
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = f"{args.device}"


def set_seed(seed):
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)
#======================================================pre.py========================================================
# model_weight_path = "./weights20220429/model-5.pth"
# mask_path = os.getcwd()+'/mask.npy'

def todense(adata):
    import scipy
    if isinstance(adata.X, scipy.sparse.csr_matrix) or isinstance(adata.X, scipy.sparse.csc_matrix):
        return adata.X.todense()
    else:
        return adata.X


def get_weight(att_mat, pathway):
    att_mat = torch.stack(att_mat).squeeze(1)
    # Average the attention weights across all heads.
    att_mat = torch.mean(att_mat, dim=1)
    # To account for residual connections, we add an identity matrix to the
    # attention matrix and re-normalize the weights.
    residual_att = torch.eye(att_mat.size(1))
    aug_att_mat = att_mat + residual_att
    aug_att_mat = aug_att_mat / aug_att_mat.sum(dim=-1).unsqueeze(-1)
    # Recursively multiply the weight matrices
    joint_attentions = torch.zeros(aug_att_mat.size())
    joint_attentions[0] = aug_att_mat[0]

    for n in range(1, aug_att_mat.size(0)):
        joint_attentions[n] = torch.matmul(aug_att_mat[n], joint_attentions[n - 1])

    # Attention from the output token to the input space.
    v = joint_attentions[-1]
    v = pd.DataFrame(v[0, 1:].detach().numpy()).T
    # print(v.size())
    v.columns = pathway
    return v


def prediect(adata, model_weight_path, project, mask_path, laten=False, save_att='X_att', save_lantent='X_lat',
             n_step=10000, cutoff=0.1, n_unannotated=1, batch_size=50, embed_dim=48, depth=2, num_heads=4, project_path = None):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    num_genes = adata.shape[1]
    # mask_path = os.getcwd()+project+'/mask.npy'
    mask = np.load(mask_path)
    # project_path = '/nfs/blanche/sy/alpine/sy/cellxgenedata/working_version/working_version/src/compare_result/TOSICA/script/result_path' + '/%s' % project
    pathway = pd.read_csv(project_path + '/pathway.csv', index_col=0)
    dictionary = pd.read_table(project_path + '/label_dictionary.csv', sep=',', header=0, index_col=0)
    n_c = len(dictionary)
    label_name = dictionary.columns[0]
    dictionary.loc[(dictionary.shape[0])] = 'Unknown'
    dic = {}
    for i in range(len(dictionary)):
        dic[i] = dictionary[label_name][i]
    model = create_model(num_classes=n_c, num_genes=num_genes, mask=mask, has_logits=False, depth=depth,
                         num_heads=num_heads).to(device)
    # load model weights
    model.load_state_dict(torch.load(model_weight_path, map_location=device))
    model.eval()
    parm = {}
    for name, parameters in model.named_parameters():
        print(name,':',parameters.size())
        parm[name] = parameters.detach().cpu().numpy()
    gene2token = parm['feature_embed.fe.weight']
    gene2token = gene2token.reshape((int(gene2token.shape[0] / embed_dim), embed_dim, adata.shape[1]))
    gene2token = abs(gene2token)
    gene2token = np.max(gene2token, axis=1)
    gene2token = pd.DataFrame(gene2token)
    gene2token.columns = adata.var_names
    gene2token.index = pathway['0']
    gene2token.to_csv(project_path + '/gene2token_weights.csv')
    latent = torch.empty([0, embed_dim]).cpu()
    att = torch.empty([0, (len(pathway))]).cpu()
    predict_class = np.empty(shape=0)
    pre_class = np.empty(shape=0)
    latent = torch.squeeze(latent).cpu().numpy()
    l_p = np.c_[latent, predict_class, pre_class]
    att = np.c_[att, predict_class, pre_class]
    all_line = adata.shape[0]
    n_line = 0
    adata_list = []
    while (n_line) <= all_line:
        if (all_line - n_line) % batch_size != 1:
            expdata = pd.DataFrame(todense(adata[n_line:n_line + min(n_step, (all_line - n_line))]), index=np.array(
                adata[n_line:n_line + min(n_step, (all_line - n_line))].obs_names).tolist(),
                                   columns=np.array(adata.var_names).tolist())
            print(n_line)
            n_line = n_line + n_step
        else:
            expdata = pd.DataFrame(todense(adata[n_line:n_line + min(n_step, (all_line - n_line - 2))]), index=np.array(
                adata[n_line:n_line + min(n_step, (all_line - n_line - 2))].obs_names).tolist(),
                                   columns=np.array(adata.var_names).tolist())
            n_line = (all_line - n_line - 2)
            print(n_line)
        expdata = np.array(expdata)
        expdata = torch.from_numpy(expdata.astype(np.float32))
        data_loader = torch.utils.data.DataLoader(expdata,
                                                  batch_size=batch_size,
                                                  shuffle=False,
                                                  pin_memory=True)
        with torch.no_grad():
            # predict class
            for step, data in enumerate(data_loader):
                # print(step)
                exp = data
                lat, pre, weights = model(exp.to(device))
                pre = torch.squeeze(pre).cpu()
                pre = F.softmax(pre, 1)
                predict_class = np.empty(shape=0)
                pre_class = np.empty(shape=0)
                for i in range(len(pre)):
                    if torch.max(pre, dim=1)[0][i] >= cutoff:
                        predict_class = np.r_[predict_class, torch.max(pre, dim=1)[1][i].numpy()]
                    else:
                        predict_class = np.r_[predict_class, n_c]
                    pre_class = np.r_[pre_class, torch.max(pre, dim=1)[0][i]]
                l_p = torch.squeeze(lat).cpu().numpy()
                att = torch.squeeze(weights).cpu().numpy()
                meta = np.c_[predict_class, pre_class]
                meta = pd.DataFrame(meta)
                meta.columns = ['Prediction', 'Probability']
                meta.index = meta.index.astype('str')
                if laten:
                    l_p = l_p.astype('float32')
                    new = sc.AnnData(l_p, obs=meta)
                else:
                    att = att[:, 0:(len(pathway) - n_unannotated)]
                    att = att.astype('float32')
                    varinfo = pd.DataFrame(pathway.iloc[0:len(pathway) - n_unannotated, 0].values,
                                           index=pathway.iloc[0:len(pathway) - n_unannotated, 0],
                                           columns=['pathway_index'])
                    new = sc.AnnData(att, obs=meta, var=varinfo)
                adata_list.append(new)
    new_adata = ad.concat(adata_list)
    new_adata.obs.index = adata.obs.index
    new_adata.obs['Prediction'] = new_adata.obs['Prediction'].map(dic)
    new_adata.obs[adata.obs.columns] = adata.obs[adata.obs.columns].values
    return (new_adata)


#======================================================train.py========================================================


class MyDataSet(Dataset):
    """
    Preproces input matrix and labels.

    """

    def __init__(self, exp, label):
        self.exp = exp
        self.label = label
        self.len = len(label)

    def __getitem__(self, index):
        return self.exp[index], self.label[index]

    def __len__(self):
        return self.len


def balance_populations(data):
    ct_names = np.unique(data[:, -1])
    ct_counts = pd.value_counts(data[:, -1])
    max_val = min(ct_counts.max(), np.int32(2000000 / len(ct_counts)))
    balanced_data = np.empty(shape=(1, data.shape[1]), dtype=np.float32)
    for ct in ct_names:
        tmp = data[data[:, -1] == ct]
        idx = np.random.choice(range(len(tmp)), max_val)
        tmp_X = tmp[idx]
        balanced_data = np.r_[balanced_data, tmp_X]
    return np.delete(balanced_data, 0, axis=0)


def splitDataSet(adata, label_name='Celltype', tr_ratio=0.7):
    """
    Split data set into training set and test set.

    """
    label_encoder = LabelEncoder()
    el_data = pd.DataFrame(todense(adata), index=np.array(adata.obs_names).tolist(),
                           columns=np.array(adata.var_names).tolist())
    el_data[label_name] = adata.obs[label_name].astype('str')
    # el_data = pd.read_table(data_path,sep=",",header=0,index_col=0)
    genes = el_data.columns.values[:-1]
    el_data = np.array(el_data)
    # el_data = np.delete(el_data,-1,axis=1)
    el_data[:, -1] = label_encoder.fit_transform(el_data[:, -1])
    inverse = label_encoder.inverse_transform(range(0, np.max(el_data[:, -1]) + 1))
    el_data = el_data.astype(np.float32)
    el_data = balance_populations(data=el_data)
    n_genes = len(el_data[1]) - 1
    train_size = int(len(el_data) * tr_ratio)
    train_dataset, valid_dataset = torch.utils.data.random_split(el_data, [train_size, len(el_data) - train_size])
    exp_train = torch.from_numpy(np.array(train_dataset)[:, :n_genes].astype(np.float32))
    label_train = torch.from_numpy(np.array(train_dataset)[:, -1].astype(np.int64))
    exp_valid = torch.from_numpy(np.array(valid_dataset)[:, :n_genes].astype(np.float32))
    label_valid = torch.from_numpy(np.array(valid_dataset)[:, -1].astype(np.int64))
    return exp_train, label_train, exp_valid, label_valid, inverse, genes


def get_gmt(gmt):
    import pathlib
    root = pathlib.Path(__file__).parent
    gmt_files = {
        "human_gobp": [root / "resources/GO_bp.gmt"],
        "human_immune": [root / "resources/immune.gmt"],
        "human_reactome": [root / "resources/reactome.gmt"],
        "human_tf": [root / "resources/TF.gmt"],
        "mouse_gobp": [root / "resources/m_GO_bp.gmt"],
        "mouse_reactome": [root / "resources/m_reactome.gmt"],
        "mouse_tf": [root / "resources/m_TF.gmt"]
    }
    return gmt_files[gmt][0]


def read_gmt(fname, sep='\t', min_g=0, max_g=5000):
    """
    Read GMT file into dictionary of gene_module:genes.\n
    min_g and max_g are optional gene set size filters.

    Args:
        fname (str): Path to gmt file
        sep (str): Separator used to read gmt file.
        min_g (int): Minimum of gene members in gene module.
        max_g (int): Maximum of gene members in gene module.
    Returns:
        OrderedDict: Dictionary of gene_module:genes.
    """
    dict_pathway = OrderedDict()
    with open(fname) as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            val = line.split(sep)
            if min_g <= len(val[2:]) <= max_g:
                dict_pathway[val[0]] = val[2:]
    return dict_pathway


def create_pathway_mask(feature_list, dict_pathway, add_missing=1, fully_connected=True, to_tensor=False):
    """
    Creates a mask of shape [genes,pathways] where (i,j) = 1 if gene i is in pathway j, 0 else.

    Expects a list of genes and pathway dict.
    Note: dict_pathway should be an Ordered dict so that the ordering can be later interpreted.

    Args:
        feature_list (list): List of genes in single-cell dataset.
        dict_pathway (OrderedDict): Dictionary of gene_module:genes.
        add_missing (int): Number of additional, fully connected nodes.
        fully_connected (bool): Whether to fully connect additional nodes or not.
        to_tensor (False): Whether to convert mask to tensor or not.
    Returns:
        torch.tensor/np.array: Gene module mask.
    """
    assert type(dict_pathway) == OrderedDict
    p_mask = np.zeros((len(feature_list), len(dict_pathway)))
    pathway = list()
    for j, k in enumerate(dict_pathway.keys()):
        pathway.append(k)
        for i in range(p_mask.shape[0]):
            if feature_list[i] in dict_pathway[k]:
                p_mask[i, j] = 1.
    if add_missing:
        n = 1 if type(add_missing) == bool else add_missing
        # Get non connected genes
        if not fully_connected:
            idx_0 = np.where(np.sum(p_mask, axis=1) == 0)
            vec = np.zeros((p_mask.shape[0], n))
            vec[idx_0, :] = 1.
        else:
            vec = np.ones((p_mask.shape[0], n))
        p_mask = np.hstack((p_mask, vec))
        for i in range(n):
            x = 'node %d' % i
            pathway.append(x)
    if to_tensor:
        p_mask = torch.Tensor(p_mask)
    return p_mask, np.array(pathway)


def train_one_epoch(model, optimizer, data_loader, device, epoch):
    """
    Train the model and updata weights.
    """
    model.train()
    loss_function = torch.nn.CrossEntropyLoss()
    accu_loss = torch.zeros(1).to(device)
    accu_num = torch.zeros(1).to(device)
    optimizer.zero_grad()
    sample_num = 0
    data_loader = tqdm(data_loader)
    for step, data in enumerate(data_loader):
        exp, label = data
        sample_num += exp.shape[0]
        _, pred, _ = model(exp.to(device))
        pred_classes = torch.max(pred, dim=1)[1]
        accu_num += torch.eq(pred_classes, label.to(device)).sum()
        loss = loss_function(pred, label.to(device))
        loss.backward()
        accu_loss += loss.detach()
        data_loader.desc = "[train epoch {}] loss: {:.3f}, acc: {:.3f}".format(epoch,
                                                                               accu_loss.item() / (step + 1),
                                                                               accu_num.item() / sample_num)
        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)
        optimizer.step()
        optimizer.zero_grad()
    return accu_loss.item() / (step + 1), accu_num.item() / sample_num


@torch.no_grad()
def evaluate(model, data_loader, device, epoch):
    model.eval()
    loss_function = torch.nn.CrossEntropyLoss()
    accu_num = torch.zeros(1).to(device)
    accu_loss = torch.zeros(1).to(device)
    sample_num = 0
    data_loader = tqdm(data_loader)
    for step, data in enumerate(data_loader):
        exp, labels = data
        sample_num += exp.shape[0]
        _, pred, _ = model(exp.to(device))
        pred_classes = torch.max(pred, dim=1)[1]
        accu_num += torch.eq(pred_classes, labels.to(device)).sum()
        loss = loss_function(pred, labels.to(device))
        accu_loss += loss
        data_loader.desc = "[valid epoch {}] loss: {:.3f}, acc: {:.3f}".format(epoch,
                                                                               accu_loss.item() / (step + 1),
                                                                               accu_num.item() / sample_num)
    return accu_loss.item() / (step + 1), accu_num.item() / sample_num


def fit_model(adata, gmt_path, project=None, pre_weights='', label_name='Celltype', max_g=300, max_gs=300,
              mask_ratio=0.015, n_unannotated=1, batch_size=8, embed_dim=48, depth=2, num_heads=4, lr=0.001, epochs=10,
              lrf=0.01, project_path=None):
    GLOBAL_SEED = 1
    set_seed(GLOBAL_SEED)
    device = 'cuda:0'
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    print(device)
    today = time.strftime('%Y%m%d', time.localtime(time.time()))
    # train_weights = os.getcwd()+"/weights%s"%today
    project = project or gmt_path.replace('.gmt', '') + '_%s' % today
    # project_path = project_path + '/%s' % project
    # if os.path.exists(project_path) is False:
    #     os.makedirs(project_path)
    tb_writer = SummaryWriter()
    exp_train, label_train, exp_valid, label_valid, inverse, genes = splitDataSet(adata, label_name)
    if gmt_path is None:
        mask = np.random.binomial(1, mask_ratio, size=(len(genes), max_gs))
        pathway = list()
        for i in range(max_gs):
            x = 'node %d' % i
            pathway.append(x)
        print('Full connection!')
    else:
        if '.gmt' in gmt_path:
            gmt_path = gmt_path
        else:
            gmt_path = get_gmt(gmt_path)
        reactome_dict = read_gmt(gmt_path, min_g=0, max_g=max_g)
        mask, pathway = create_pathway_mask(feature_list=genes,
                                            dict_pathway=reactome_dict,
                                            add_missing=n_unannotated,
                                            fully_connected=True)
        pathway = pathway[np.sum(mask, axis=0) > 4]
        mask = mask[:, np.sum(mask, axis=0) > 4]
        # print(mask.shape)
        pathway = pathway[sorted(np.argsort(np.sum(mask, axis=0))[-min(max_gs, mask.shape[1]):])]
        mask = mask[:, sorted(np.argsort(np.sum(mask, axis=0))[-min(max_gs, mask.shape[1]):])]
        # print(mask.shape)
        print('Mask loaded!')
    np.save(project_path + '/mask.npy', mask)
    pd.DataFrame(pathway).to_csv(project_path + '/pathway.csv')
    pd.DataFrame(inverse, columns=[label_name]).to_csv(project_path + '/label_dictionary.csv', quoting=None)
    num_classes = np.int64(torch.max(label_train) + 1)
    # print(num_classes)
    train_dataset = MyDataSet(exp_train, label_train)
    valid_dataset = MyDataSet(exp_valid, label_valid)
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               pin_memory=True, drop_last=True)
    valid_loader = torch.utils.data.DataLoader(valid_dataset,
                                               batch_size=batch_size,
                                               shuffle=False,
                                               pin_memory=True, drop_last=True)
    model = create_model(num_classes=num_classes, num_genes=len(exp_train[0]), mask=mask, embed_dim=embed_dim,
                         depth=depth, num_heads=num_heads, has_logits=False).to(device)
    if pre_weights != "":
        assert os.path.exists(pre_weights), "pre_weights file: '{}' not exist.".format(pre_weights)
        preweights_dict = torch.load(pre_weights, map_location=device)
        print(model.load_state_dict(preweights_dict, strict=False))

    print('Model builded!')
    pg = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(pg, lr=lr, momentum=0.9, weight_decay=5E-5)
    lf = lambda x: ((1 + math.cos(x * math.pi / epochs)) / 2) * (1 - lrf) + lrf
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    for epoch in range(epochs):
        train_loss, train_acc = train_one_epoch(model=model,
                                                optimizer=optimizer,
                                                data_loader=train_loader,
                                                device=device,
                                                epoch=epoch)
        scheduler.step()
        val_loss, val_acc = evaluate(model=model,
                                     data_loader=valid_loader,
                                     device=device,
                                     epoch=epoch)
        tags = ["train_loss", "train_acc", "val_loss", "val_acc", "learning_rate"]
        tb_writer.add_scalar(tags[0], train_loss, epoch)
        tb_writer.add_scalar(tags[1], train_acc, epoch)
        tb_writer.add_scalar(tags[2], val_loss, epoch)
        tb_writer.add_scalar(tags[3], val_acc, epoch)
        tb_writer.add_scalar(tags[4], optimizer.param_groups[0]["lr"], epoch)

    
    torch.save(model.state_dict(), project_path + "/model.pth")
    print('Training finished!')

# train(adata, gmt_path, pre_weights, batch_size=8, epochs=20)


#======================================================run_TOSICA.py========================================================


def stratified_subsample(adata, label_key='cell_type', n_cells=100000, random_state=42):
    np.random.seed(random_state)
    obs_df = adata.obs.copy()
    obs_df['idx'] = np.arange(len(adata))

    # Ensure at least one cell per type is sampled
    grouped = obs_df.groupby(label_key)
    guaranteed_indices = grouped.apply(lambda g: g.sample(n=1, random_state=random_state))['idx'].values

    n_remaining = n_cells - len(guaranteed_indices)
    if n_remaining < 0:
        raise ValueError(f"Requested {n_cells} cells, but there are {len(guaranteed_indices)} unique labels. Increase n_cells.")

    # Get remaining pool excluding already selected indices
    remaining_df = obs_df[~obs_df['idx'].isin(guaranteed_indices)]
    
    # Sample the remaining cells randomly (optionally weighted)
    additional_indices = remaining_df.sample(n=n_remaining, random_state=random_state)['idx'].values

    # Combine and subset
    final_indices = np.concatenate([guaranteed_indices, additional_indices])
    return adata[final_indices].copy()


def train(adata, gmt_path, project=None, pre_weights='', label_name='Celltype', max_g=300, max_gs=300, mask_ratio=0.015, n_unannotated=1, batch_size=8, embed_dim=48, depth=2, num_heads=4, lr=0.001, epochs=10, lrf=0.01, project_path=None):
    fit_model(adata, gmt_path, project=project, pre_weights=pre_weights, label_name=label_name,
              max_g=max_g, max_gs=max_gs, mask_ratio=mask_ratio, n_unannotated=n_unannotated, batch_size=batch_size,
              embed_dim=embed_dim, depth=depth, num_heads=num_heads, lr=lr, epochs=epochs, lrf=lrf, project_path=project_path)

def pre(adata, model_weight_path, project, laten=True, save_att='X_att', save_lantent='X_lat', n_step=10000, cutoff=0.1, n_unannotated=1, batch_size=50, embed_dim=48, depth=2, num_heads=4, project_path=None):
    mask_path = project_path + '/mask.npy'
    adata = prediect(adata, model_weight_path, project=project, mask_path=mask_path, laten=laten,
                     save_att=save_att, save_lantent=save_lantent, n_step=n_step, cutoff=cutoff, n_unannotated=n_unannotated, batch_size=batch_size, embed_dim=embed_dim, depth=depth, num_heads=num_heads, project_path=project_path)
    return adata

def run_tosica(dts_name, path_train=None, path_test=None, path_to_save= None):
    print('Running dataset:', dts_name)
    path_to_weights = f'{path_to_save}/weights/'

    os.makedirs(path_to_save, exist_ok=True)
    os.makedirs(path_to_weights, exist_ok=True)

    start_time = time.time()


    data_train = sc.read_h5ad(path_train)
    data_test = sc.read_h5ad(path_test)

    label_key = 'cell_type_id'
    ## set var_names to gene symbols
    data_train.var_names = data_train.var['feature_name'].values
    data_test.var_names = data_test.var['feature_name'].values
    
    ## subsample the training data if has more than 500000 cells
    if data_train.n_obs > 100000:
        data_train = stratified_subsample(data_train, label_key=label_key, n_cells=100000)
    
    if data_test.n_obs > 100000:
        data_test = stratified_subsample(data_train, label_key=label_key, n_cells=100000)

   
    ###=========================
    # normalize
    if data_train.X.max() > 20:
        sc.pp.normalize_total(data_train, target_sum=1e4)
        sc.pp.log1p(data_train)
    if data_test.X.max() > 20:
        sc.pp.normalize_total(data_test, target_sum=1e4)
        sc.pp.log1p(data_test)
    ###=========================

    n_class = len(data_train.obs[label_key].unique())
    if n_class < 2:
        print(f"Number of classes in {dts_name} is less than 2. Skipping...")
        return

    ## set the number of maximum training steps based on the number of training cells, approximate number of training steps for the model to converge


    if data_train.shape[0] < 50000:
        max_steps = 1000
    else:
        max_steps = 4000

    if data_train.shape[0] < 64:
        batch_size = 16
    elif data_train.shape[0] < 1000:
        batch_size = 64
    elif data_train.shape[0] < 5000:
        batch_size = 128
    elif data_train.shape[0] < 10000:
        batch_size = 256
    else:
        batch_size = 512

    ## calculate the number of epochs based on the batch size and max_steps, make sure it is at least 10 epochs and the model converges
    epochs = max(10, max_steps // (data_train.shape[0] // batch_size))
    print(f"Number of epochs: {epochs}")
    train(data_train, gmt_path="human_gobp", label_name=label_key, epochs=epochs, batch_size=batch_size, project=dts_name, project_path=path_to_weights)

    model_weight_path = path_to_weights + f"/model.pth"
    new_adata = pre(data_test, model_weight_path=model_weight_path, project=dts_name, project_path=path_to_weights, laten = True)
    new_adata.obs["Prediction"].to_csv(f"{path_to_save}/prediction.csv")

    train_labels = data_train.obs[label_key].values
    test_label = data_test.obs[label_key].values
    pred_test_labels = new_adata.obs["Prediction"].values

    unique_train_labels = set(train_labels)
    idx2cal = [lb in unique_train_labels for lb in test_label]
    print("Number of samples in test to calculate metrics: ", sum(idx2cal))
    new_adata.obs = new_adata.obs.astype(str)
    new_adata.obs['idx2cal'] = idx2cal

    new_adata.write_h5ad(f"{path_to_save}/pred.h5ad")

    pred_test_labels, test_label = pred_test_labels[idx2cal], np.array(test_label)[idx2cal]

    acc = accuracy_score(test_label, pred_test_labels)
    f1_macro = f1_score(test_label, pred_test_labels, average='macro')
    f1_weighted = f1_score(test_label, pred_test_labels, average='weighted')
    precision = precision_score(test_label, pred_test_labels, average='macro')
    recall = recall_score(test_label, pred_test_labels, average='macro')
    total_time = time.time() - start_time

    with open(f"{path_to_save}/result.txt", "w") as f:
        f.write(f"Accuracy: {acc}\n")
        f.write(f"F1 Macro: {f1_macro}\n")
        f.write(f"F1 Weighted: {f1_weighted}\n")
        f.write(f"Precision: {precision}\n")
        f.write(f"Recall: {recall}\n")
        f.write(f"Total Time: {total_time}\n")
        f.write(f"Number of epochs: {epochs}\n")
        f.write(f"Batch size: {batch_size}\n")

    return acc, f1_macro, f1_weighted, precision, recall, total_time

def get_path_to_save(save_root, tissue):
    # Define the path to save results
    path_to_save = os.path.join(save_root, tissue)
    # Create the directory if it does not exist
    return path_to_save
#%%
if __name__ == '__main__':
    import multiprocessing as mp

    mp.set_start_method('spawn')
    set_seed(1)
    
    DATA_DIR = Path(args.data_dir)
    RESULTS_DIR = Path(args.result_dir)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    data_dir = DATA_DIR / 'cellxgene'
    tissues = os.listdir(data_dir)
    tissues = ['adrenal gland']
    
    save_root = RESULTS_DIR / "main_results" / "TOSICA" 
    save_root.mkdir(parents=True, exist_ok=True)

    os.makedirs(save_root, exist_ok=True)

    remain_tissues = []
    for tissue in tissues:
        result_file = os.path.join(f"{get_path_to_save(save_root, tissue)}", "result.txt")
        if os.path.exists(result_file):
            continue
        remain_tissues.append(tissue)
    tissues = remain_tissues
    path_train_template = "{tissue}/train.h5ad"
    path_test_template = "{tissue}/test.h5ad"
    print("Number of tissues to process: ", len(tissues))
    # create jobs for multiprocessing
    jobs = []
    for tissue in tissues:
        path_train = data_dir / path_train_template.format(tissue=tissue)
        path_test = data_dir / path_test_template.format(tissue=tissue)
        path_to_save = get_path_to_save(save_root, tissue)
        if not os.path.exists(path_train) or not os.path.exists(path_test):
            print(f"Data files for {tissue} do not exist. Skipping...")
            continue
        jobs.append((tissue, path_train, path_test, path_to_save))

    # Run all jobs sequentially
    print("Using sequential processing")
    for tissue, path_train, path_test, path_to_save in jobs:
        run_tosica(tissue, path_train, path_test, path_to_save)
