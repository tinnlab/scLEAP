# Reproduce scLMCT foundation model

# Installation
We use merlin loader for faster training. To install all the required packages: 

```conda create -n merlin -y```
```conda activate merlin```
```conda install -c nvidia -c rapidsai -c numba -c conda-forge merlin-core merlin-dataloader cudf nvtabular python cudatoolkit ipykernel cudatoolkit-dev numpy==1.26.4 -y```

Install scLMCT in development mode:
```pip install -e .```

## Training the model
```python train_foundation_model.py --seed 1 --gpu 0 --data_dir "./data/foundation-training-data"```