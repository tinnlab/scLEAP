import torch
from torch.utils.data import Dataset, DataLoader
import anndata
import numpy as np
import torch.nn as nn


class RandomGeneMask(torch.nn.Module):
    def __init__(self, mask_prob=0.1):
        super().__init__()
        self.mask_prob = mask_prob
        
    def forward(self, x):
        if not self.training:  # Only augment during training!
            return x
        mask = torch.bernoulli(torch.full_like(x, 1 - self.mask_prob))
        return x * mask

class AddGaussianNoise(torch.nn.Module):
    def __init__(self, std=0.1):
        super().__init__()
        self.std = std
        
    def forward(self, x):
        if not self.training:
            return x
        noise = torch.randn_like(x) * self.std
        return x + noise
    
class RandomLibrarySizeScaling(torch.nn.Module):
    def __init__(self, scale_min=0.8, scale_max=1.2):
        super().__init__()
        self.scale_min = scale_min
        self.scale_max = scale_max
        
    def forward(self, x):
        if not self.training:
            return x
        scale = torch.empty(x.shape[0], 1, device=x.device).uniform_(self.scale_min, self.scale_max)
        return x * scale


class RandomGeneShuffle(torch.nn.Module):
    def __init__(self, n_genes=10):
        super().__init__()
        self.n_genes = n_genes
        
    def forward(self, x):
        if not self.training or self.n_genes <= 0:
            return x
        x = x.clone()
        for i in range(x.shape[0]):  # for each cell
            idx = torch.randperm(x.shape[1])[:self.n_genes]
            shuffled = x[i, idx][torch.randperm(self.n_genes)]
            x[i, idx] = shuffled
        return x


class LOG1PTransform(nn.Module):
    def forward(self, x):
        return torch.log1p(x)

class TotalSumNormalize(nn.Module):
    def __init__(self, target_sum=1.0, eps=1e-8):
        super().__init__()
        self.target_sum = target_sum
        self.eps = eps

    def forward(self, x):
        total = x.sum(dim=-1, keepdim=True)
        scale = torch.where(total < self.eps, torch.ones_like(total), self.target_sum / (total + 1e-8))
        return x * scale

    
class H5ADDataset(Dataset):
    """
    PyTorch Dataset for loading data from H5AD (AnnData) files.
    """
    def __init__(self, file_path, transform=None, target_transform=None, 
                 use_obs_column=None, use_highly_variable=False, 
                 use_sparse=False, return_obs_names=False, ols_mappings = None):
        """
        Initialize the H5AD Dataset.
        
        Parameters:
        -----------
        file_path : str
            Path to the H5AD file
        transform : callable, optional
            Optional transform to be applied to the features
        target_transform : callable, optional
            Optional transform to be applied to the target labels
        use_obs_column : str, optional
            The column in .obs to use as labels. If None, assumes no labels are available
        use_highly_variable : bool, default=False
            If True, only use highly variable genes if they are annotated in the dataset
        use_sparse : bool, default=False
            If True, return sparse matrices instead of dense arrays
        return_obs_names : bool, default=False
            If True, return cell names/identifiers from .obs_names
        """
        self.transform = transform
        self.target_transform = target_transform
        self.use_obs_column = use_obs_column
        self.use_highly_variable = use_highly_variable
        self.use_sparse = use_sparse
        self.return_obs_names = return_obs_names
        
        # Load the AnnData object
        self.adata = anndata.read_h5ad(file_path)
        
        # Use highly variable genes if specified and available
        if use_highly_variable and 'highly_variable' in self.adata.var:
            self.hvg_indices = np.where(self.adata.var['highly_variable'])[0]
        else:
            self.hvg_indices = None
            
        # Extract labels if specified
        if self.use_obs_column and self.use_obs_column in self.adata.obs:
            if ols_mappings is not None:
                label_series = self.adata.obs[self.use_obs_column].str.lower().replace("_", " ").apply(lambda x: ols_mappings.get(x, -1))
                if -1 in label_series.values:
                    print(f"Warning: Some labels in '{self.use_obs_column}' were not found in the provided mappings. Using original labels.")
                    label_series = self.adata.obs[self.use_obs_column]
            else:
                label_series = self.adata.obs[self.use_obs_column]

            # Convert to numeric if categorical
            if hasattr(label_series, 'cat'):
                sorted_categories = sorted(label_series.cat.categories)
                self.label_dict = {cat: i for i, cat in enumerate(sorted_categories)}
                self.index_to_label = {i: cat for i, cat in enumerate(sorted_categories)}
                self.labels = np.array([self.label_dict[cat] for cat in label_series])
            else:
                unique_labels = np.unique(label_series)
                sorted_labels = sorted(unique_labels)
                self.label_dict = {label: i for i, label in enumerate(sorted_labels)}
                self.index_to_label = {i: label for i, label in enumerate(sorted_labels)}
                self.labels = np.array([self.label_dict[label] for label in label_series])

        else:
            self.labels = None
            self.label_dict = None
            self.index_to_label = None
            
    def __len__(self):
        """Return the number of cells/samples in the dataset"""
        return self.adata.n_obs
    
    def __getitem__(self, idx):
        """
        Get a single data point from the dataset
        
        Parameters:
        -----------
        idx : int
            Index of the data point
            
        Returns:
        --------
        tuple or features
            If use_obs_column is provided, returns (features, label)
            Otherwise, just returns features
        """
        # Get features (gene expression)
        if self.use_sparse and isinstance(self.adata.X, np.ndarray):
            features = self.adata.X[idx].copy()
        elif not self.use_sparse and not isinstance(self.adata.X, np.ndarray):
            features = self.adata.X[idx].toarray().flatten()
        else:
            if isinstance(self.adata.X, np.ndarray):
                features = self.adata.X[idx].copy()
            else:
                features = self.adata.X[idx].toarray().flatten()
                
        # Use only highly variable genes if specified
        if self.hvg_indices is not None:
            features = features[self.hvg_indices]
        
        
        
        # Convert to PyTorch tensor
        if isinstance(features, np.ndarray):
            features = torch.FloatTensor(features)
        else:  # For sparse matrices
            features = torch.FloatTensor(features.toarray().flatten())
        
        # Apply transforms if provided
        if self.transform:
            features = self.transform(features)
        # Return data based on configuration
        if self.use_obs_column and self.labels is not None:
            label = self.labels[idx]
            if self.target_transform:
                label = self.target_transform(label)
            label = torch.tensor(label, dtype=torch.long)
            
            if self.return_obs_names:
                return features, label, self.adata.obs_names[idx]
            else:
                return features, label
        else:
            if self.return_obs_names:
                return features, self.adata.obs_names[idx]
            else:
                return features
                
    def get_class_weights(self):
        """
        Calculate class weights for imbalanced datasets
        
        Returns:
        --------
        weights : torch.Tensor
            Tensor of class weights
        """
        if self.labels is None:
            raise ValueError("No labels available to calculate class weights")
        
        class_counts = np.bincount(self.labels.astype(int))
        weights = 1.0 / class_counts
        weights = weights / weights.sum() * len(weights)  # Normalize
        return torch.FloatTensor(weights)
    
    def get_label_mapping(self):
        """
        Returns the mapping from categorical labels to numeric indices
        
        Returns:
        --------
        dict or None
            Dictionary mapping categories to indices, or None if no mapping exists
        """
        if hasattr(self, 'label_dict'):
            return self.label_dict
        return None
    
    def get_index_to_label_mapping(self):
        """
        Returns the mapping from numeric indices to categorical labels
        
        Returns:
        --------
        dict or None
            Dictionary mapping indices to original categories, or None if no mapping exists
        """
        if hasattr(self, 'index_to_label'):
            return self.index_to_label
        return None
    
    def convert_predictions_to_labels(self, predictions):
        """
        Convert numeric predictions back to original categorical labels
        
        Parameters:
        -----------
        predictions : array-like
            Array of numeric predictions (class indices)
            
        Returns:
        --------
        list
            List of original categorical labels
        """
        if self.index_to_label is None:
            raise ValueError("No label mapping available for conversion")
        
        # Convert tensor to numpy if needed
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.cpu().numpy()
        
        # Convert indices to original labels
        return [self.index_to_label[int(pred)] for pred in predictions]
    
    def get_features_dim(self):
        """
        Returns the dimension of the features
        
        Returns:
        --------
        int
            Dimension of feature vectors
        """
        if self.hvg_indices is not None:
            return len(self.hvg_indices)
        return self.adata.n_vars

# Example usage:
if __name__ == "__main__":
    # Define paths
    tissue = "eye"
    PATH_TRAIN = f"/data/share/sy/cell_classification/othermethod_data/data_updated/data_v3/bone spine/run_1/train.h5ad"
    PATH_TEST = f"/data/share/sy/cell_classification/othermethod_data/data_updated/data_v3/{tissue}/run_1/test.h5ad"
    
    # Create datasets
    train_dataset = H5ADDataset(
        file_path=PATH_TRAIN,
        use_obs_column="cell_type",  # Assumes cell type is in this column
        use_highly_variable=True  # Use only highly variable genes if annotated
    )
    
    test_dataset = H5ADDataset(
        file_path=PATH_TEST,
        use_obs_column="cell_type",
        use_highly_variable=True
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=32,
        shuffle=True,
        num_workers=4
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=32,
        shuffle=False,
        num_workers=4
    )
    
    # Example of model training and prediction (simplified)
    # Assuming you have a trained model that outputs class probabilities
    dummy_predictions = torch.tensor([0, 1, 2, 1, 0, 3])
    
    # Convert numeric predictions back to original cell type labels
    original_labels = test_dataset.convert_predictions_to_labels(dummy_predictions)
    print(f"Numeric predictions: {dummy_predictions}")
    print(f"Original cell type labels: {original_labels}")
    
    # Print the mapping for reference
    print("\nClass index to label mapping:")
    for idx, label in test_dataset.get_index_to_label_mapping().items():
        print(f"  {idx}: {label}")