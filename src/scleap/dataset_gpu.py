"""
GPU-based dataset that loads sparse data to GPU using CuPy.
"""

import torch
from torch.utils.data import Dataset
import anndata
import numpy as np
import cupy as cp
import cupyx.scipy.sparse as cpx_sparse


class H5ADDatasetGPU(Dataset):
    """
    Dataset that loads sparse h5ad data to GPU using CuPy.
    Eliminates per-batch CPU-GPU transfers.
    """

    def __init__(self, file_path, use_obs_column=None, transform=None,
                 device='cuda', ols_mappings=None):
        """
        Load sparse dataset to GPU.

        Parameters:
        -----------
        file_path : str
            Path to h5ad file
        use_obs_column : str
            Column name for labels
        transform : callable
            Transform to apply (will be applied on GPU)
        device : str
            GPU device (e.g., 'cuda:0')
        ols_mappings : dict
            Label mappings
        """
        import os
        file_size_gb = os.path.getsize(file_path) / 1024**3
        print(f"Loading dataset to {device}...")
        print(f"  H5AD file size: {file_size_gb:.2f} GB")

        # Load h5ad file
        print(f"  Reading h5ad file from disk...")
        adata = anndata.read_h5ad(file_path)
        print(f"  Loaded: {adata.X.shape}, sparsity={1 - adata.X.nnz / (adata.X.shape[0] * adata.X.shape[1]):.3f}")

        # Get sparse matrix
        X_sparse = adata.X
        print(f"  Matrix format: {X_sparse.format}, dtype={X_sparse.dtype}")

        # Calculate transfer size
        total_bytes = X_sparse.data.nbytes + X_sparse.indices.nbytes + X_sparse.indptr.nbytes
        print(f"  Transferring to GPU: {len(X_sparse.data):,} non-zero values, {total_bytes / 1024**3:.3f} GB")

        # Transfer to GPU as CuPy sparse matrix
        print(f"  Copying to GPU memory...")
        X_gpu_sparse = cpx_sparse.csr_matrix(X_sparse, dtype=cp.float32)

        self.X_gpu = X_gpu_sparse
        gpu_bytes = self.X_gpu.data.nbytes + self.X_gpu.indices.nbytes + self.X_gpu.indptr.nbytes
        print(f"  ✓ Data on GPU: {gpu_bytes / 1024**3:.3f} GB (dtype=float32)")

        # Store transforms
        self.transform = transform
        if self.transform:
            self.transform = self.transform.to(device)

        self.device = device

        # Process labels
        if use_obs_column and use_obs_column in adata.obs:
            if ols_mappings is not None:
                label_series = adata.obs[use_obs_column].str.lower().replace("_", " ").apply(
                    lambda x: ols_mappings.get(x, -1))
            else:
                label_series = adata.obs[use_obs_column]

            # Convert to numeric
            if hasattr(label_series, 'cat'):
                sorted_categories = sorted(label_series.cat.categories)
                self.label_dict = {cat: i for i, cat in enumerate(sorted_categories)}
                self.index_to_label = {i: cat for i, cat in enumerate(sorted_categories)}
                labels = np.array([self.label_dict[cat] for cat in label_series])
            else:
                unique_labels = np.unique(label_series)
                sorted_labels = sorted(unique_labels)
                self.label_dict = {label: i for i, label in enumerate(sorted_labels)}
                self.index_to_label = {i: label for i, label in enumerate(sorted_labels)}
                labels = np.array([self.label_dict[label] for label in label_series])

            # Labels to GPU
            self.labels_gpu = torch.LongTensor(labels).to(device)
        else:
            self.labels_gpu = None
            self.label_dict = None
            self.index_to_label = None

        # Clean up CPU memory
        del adata

        print(f"GPU dataset ready: {len(self)} samples on {device}")

    def __len__(self):
        return self.X_gpu.shape[0]

    def __getitem__(self, idx):
        """Single sample access"""
        row_sparse = self.X_gpu[idx]
        row_dense_cupy = row_sparse.toarray().ravel()
        features = torch.as_tensor(row_dense_cupy, device=self.device)

        if self.transform:
            features = self.transform(features)

        if self.labels_gpu is not None:
            label = self.labels_gpu[idx]
            return features, label
        else:
            return features

    def __getitems__(self, indices):
        """Batch indexing - efficient for GPU sparse matrices"""
        # Convert list to slice or array for efficient indexing
        if isinstance(indices, list):
            if len(indices) > 1 and all(indices[i] == indices[0] + i for i in range(len(indices))):
                # Contiguous - use slice
                batch_sparse = self.X_gpu[indices[0]:indices[-1]+1]
            else:
                # Non-contiguous
                indices_gpu = cp.array(indices)
                batch_sparse = self.X_gpu[indices_gpu]
        else:
            batch_sparse = self.X_gpu[indices]

        # Convert entire batch to dense on GPU
        batch_dense_cupy = batch_sparse.toarray()
        batch_features = torch.as_tensor(batch_dense_cupy, device=self.device)

        # Apply transforms
        if self.transform:
            batch_features = self.transform(batch_features)

        # Get labels
        if self.labels_gpu is not None:
            if isinstance(indices, list):
                batch_labels = self.labels_gpu[indices]
            else:
                batch_labels = self.labels_gpu[indices]

            return [(batch_features[i], batch_labels[i]) for i in range(len(batch_features))]
        else:
            return [batch_features[i] for i in range(len(batch_features))]

    def get_label_mapping(self):
        return self.label_dict

    def get_index_to_label_mapping(self):
        return self.index_to_label
