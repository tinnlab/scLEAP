import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from scleap.loss import LabelAwareMarginLoss, RepelContrastiveLoss, PermutedContrastiveLoss, GraphLoss

from typing import List, Dict
import os

import torch
import transformers
from transformers import AutoTokenizer, AutoModel  

class ResidualBlock(nn.Module):
	def __init__(self, in_features, out_features):
		super().__init__()
		self.lin1 = nn.Linear(in_features, out_features)
		self.bn1 = nn.BatchNorm1d(out_features)
		self.lin2 = nn.Linear(out_features, out_features)
		self.bn2 = nn.BatchNorm1d(out_features)
		self.activation = nn.GELU()
		self.dropout = nn.Dropout(0.2)

		if in_features != out_features:
			self.shortcut = nn.Sequential(
				nn.Linear(in_features, out_features),
				nn.BatchNorm1d(out_features)
			)
		else:
			self.shortcut = nn.Identity()

		self.res_scale = 0.5  # key to prevent magnitude blow-up

	def forward(self, x):
		residual = self.shortcut(x)
		out = self.activation(self.bn1(self.lin1(x)))
		out = self.dropout(out)
		out = self.bn2(self.lin2(out))
		out = out * self.res_scale + residual
		out = self.activation(out)
		out = torch.clamp(out, -6, 6)  # prevent float16 overflow
		return out
	
class CellEncoder(nn.Module):
	"""
	CellEncoder is a neural network module that encodes input features into a higher-dimensional space.
	It consists of multiple residual blocks. The last block maps to output_dim.
	"""
	def __init__(self, input_dim, hidden_dim, output_dim, n_layers=3, use_linear_out=False):
		super(CellEncoder, self).__init__()
		self.input_dim = input_dim
		self.hidden_dim = hidden_dim
		self.output_dim = output_dim
		self.n_layers = n_layers

		self.use_linear_out = use_linear_out
		self.linear_out = nn.Linear(output_dim, output_dim, bias=False)
		# Create residual blocks
		self.residual_blocks = nn.ModuleList()
		for i in range(n_layers):
			in_dim = input_dim if i == 0 else hidden_dim
			out_dim = output_dim if i == n_layers - 1 else hidden_dim

			self.residual_blocks.append(ResidualBlock(in_dim, out_dim))
	def forward(self, x):
		"""
		Forward pass through the CellEncoder.
		Args:
			x (torch.Tensor): Input tensor of shape (batch_size, input_dim).
		Returns:
			torch.Tensor: Encoded output tensor of shape (batch_size, output_dim).
		"""
		for block in self.residual_blocks:
			x = block(x)
		if self.use_linear_out:
			x = self.linear_out(x)
		return x

class TextEncoderWrapper(nn.Module):
	def __init__(self,
				 model_name: str = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext",
				 device: str = "cuda",
				 cache_path="cached_prompt_embeddings.pt",
				 n_text_layers_to_finetune: int = 1,
				 use_projection_head: bool = False,
				 projection_dim: int = None):

		"""
		TextEncoderWrapper wraps a pre-trained transformer model for text encoding.
		params:
			model_name: name of the pre-trained model from HuggingFace
			device: device to run the model on
			cache_path: path to cache prompt embeddings	
			n_text_layers_to_finetune: number of top transformer layers to finetune
			use_projection_head: whether to use a projection head after the transformer
			projection_dim: dimension of the projection head output if used
		"""
		super().__init__()
		self.device = device
		self.model_name = model_name
		self.tokenizer = AutoTokenizer.from_pretrained(model_name)
		load_kwargs = {}
		torch_version = tuple(int(part) for part in torch.__version__.split("+", 1)[0].split(".")[:2])
		if torch_version < (2, 6):
			# transformers>=4.55 rejects torch.load-based checkpoints on torch<2.6,
			# so prefer safetensors-backed model repos in older environments.
			load_kwargs["use_safetensors"] = True

		try:
			self.model = AutoModel.from_pretrained(model_name, **load_kwargs)
		except Exception as exc:
			if load_kwargs.get("use_safetensors"):
				raise ValueError(
					f"Failed to load '{model_name}' with safetensors in this environment "
					f"(torch {torch.__version__}, transformers {transformers.__version__}). "
					"Use a model repo that ships 'model.safetensors' such as "
					"'cambridgeltl/SapBERT-from-PubMedBERT-fulltext', or upgrade torch to >=2.6 "
					"if you need to load .bin-only checkpoints."
				) from exc
			raise
		self.cache_path = cache_path
		self.prompt_layer_cache = {}

		# Validate model structure
		assert hasattr(self.model, "encoder") and hasattr(self.model.encoder, "layer"), \
			"Provided model does not have BERT-style encoder layers"
		self.num_layers = len(self.model.encoder.layer)
		print(f"[INFO] Loaded model with {self.num_layers} transformer layers.")

		# Freeze all layers by default
		for param in self.model.parameters():
			param.requires_grad = False
		

		self.model.to(device)

		# Projection head setup
		self.use_projection_head = use_projection_head
		self.projection_dim = projection_dim
		hidden_size = self.model.config.hidden_size

		if projection_dim is None:
			print("[INFO] No projection_dim provided, using hidden_size as projection_dim.")
			self.projection_dim = hidden_size  
			
		else:
			self.projection_dim = projection_dim  

		if use_projection_head:
			self.text_proj = nn.Sequential(
				nn.Linear(hidden_size, hidden_size),
				nn.GELU(),
				nn.Linear(hidden_size, projection_dim)
			)
		else:
			self.text_proj = nn.Identity()

		if n_text_layers_to_finetune < self.num_layers and n_text_layers_to_finetune > 0:
			for name, param in self.model.named_parameters():
				for i in range(1, n_text_layers_to_finetune + 1):
					if f"encoder.layer.{self.num_layers - i}" in name:
						param.requires_grad = True


	def tokenize_prompts(self, label_prompts, max_length=256):
		return {
			class_id: self.tokenizer(
				prompts, padding=True, truncation=True, max_length=max_length, return_tensors="pt"
			)
			for class_id, prompts in label_prompts.items()
		}

	def load_or_generate_prompt_cache(self, prompt_token_batches):
		if os.path.exists(self.cache_path):
			print(f"[INFO] Loading prompt cache from {self.cache_path}")
			self.prompt_layer_cache = torch.load(self.cache_path, map_location=self.device)
		else:
			print("[INFO] Generating prompt cache...")
			self.model.eval()
			with torch.no_grad():
				for class_id, token_batch in prompt_token_batches.items():
					token_batch = {k: v.to(self.device) for k, v in token_batch.items()}
					input_ids = token_batch["input_ids"]
					attention_mask = token_batch["attention_mask"]

					hidden_states = self.model.embeddings(input_ids=input_ids)
					extended_attention_mask = (1.0 - attention_mask[:, None, None, :]) * -10000.0

					# Run through all layers except the last one
					for i in range(self.num_layers - 1):
						layer_module = self.model.encoder.layer[i]
						hidden_states = layer_module(hidden_states, attention_mask=extended_attention_mask)[0]

					self.prompt_layer_cache[class_id] = (hidden_states.cpu(), attention_mask.cpu())

			torch.save(self.prompt_layer_cache, self.cache_path)
			print(f"[INFO] Saved prompt cache to {self.cache_path}")

	def run_final_layer(self, hidden_states, attention_mask, pool_type='cls'):
		# Create extended attention mask for final transformer layer
		ext_mask = attention_mask[:, None, None, :].to(dtype=hidden_states.dtype)
		ext_mask = (1.0 - ext_mask) * -10000.0

		# Run final transformer layer
		output = self.model.encoder.layer[self.num_layers - 1](
			hidden_states, attention_mask=ext_mask
		)[0]  # shape: [B, T, H]

		# Apply pooling strategy
		if pool_type == 'cls':
			pooled = output[:, 0, :]  # CLS token
		elif pool_type == 'mean':
			mask = attention_mask.unsqueeze(-1).to(dtype=output.dtype)
			pooled = (output * mask).sum(dim=1) / mask.sum(dim=1)
		else:
			raise ValueError(f"Unknown pool_type: {pool_type}")

		# Apply projection (either Identity or a projection head)
		return self.text_proj(pooled)



class TrainWrapperCLIPStyle(pl.LightningModule):
	def __init__(self,
				 input_dim: int,
				 hidden_dim: int,
				 output_dim: int,
				 label_prompts: Dict[int, List[str]],
				 n_cell_layers: int = 2,
				 text_encoder_model: str = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext",
				 use_projection_head: bool = False,
				 n_prompts: int = 5,
				 n_text_layers_to_finetune: int = 1,
				 pool_type: str = 'cls',
				 device: str = 'cuda',
				 cache_path: str = "cached_prompt_embeddings.pt",
				 lr: float = 1e-4,
				 weight_decay: float = 1e-5,
				 s: float = 30.0,
				 m1: float = 1.0,
				 m2: float = 0.4,
				 m3: float = 0.2,
				 m4: float = 1.5,
				 f_gamma: float = 2.0,
				 l_ct: float = 1,
				 l_tt: float = 0.1,
				 l_contrastive: float = 0.1,
				 l_graph: float = 0.1,
				 graph_embeddings = None

				 ):

		"""
		params:
			input_dim: dimension of input features
			hidden_dim: dimension of hidden layers in CellEncoder
			output_dim: dimension of output embeddings
			label_prompts: dictionary mapping class IDs to list of prompt strings
			n_cell_layers: number of layers in CellEncoder
			text_encoder_model: name of pre-trained text encoder model
			use_projection_head: whether to use a projection head in text encoder
			n_prompts: number of prompt embeddings to use per class in each training step
			n_text_layers_to_finetune: number of top transformer layers to finetune
			pool_type: pooling strategy for text encoder ('cls' or 'mean')
			device: device to run the model on
			cache_path: path to cache prompt embeddings
			lr: learning rate
			weight_decay: weight decay for optimizer
			s, m1, m2, m3, f_gamma: parameters for LabelAwareMarginLoss
			l_lambda: weight for combining two losses 
		"""
		super().__init__()
		self.save_hyperparameters()

		self.text_encoder_wrapper = TextEncoderWrapper(model_name=text_encoder_model, device=device, cache_path=cache_path, n_text_layers_to_finetune=n_text_layers_to_finetune, use_projection_head=use_projection_head, projection_dim=output_dim)
		
		if not output_dim:
			self.output_dim = self.text_encoder_wrapper.model.config.hidden_size
		else:
			self.output_dim = output_dim

		self.encoder = CellEncoder(input_dim, hidden_dim, self.output_dim, n_layers=n_cell_layers, use_linear_out=False)

		self.label_prompts = label_prompts
		self.prompt_token_batches = {}
		self.n_prompts = n_prompts
		self.pool_type = pool_type

		self.lr = lr
		self.wd = weight_decay
		self.l_ct = l_ct
		self.l_tt = l_tt
		self.l_contrastive = l_contrastive
		self.l_graph = l_graph

		## loss functions
		self.loss_cell_text = LabelAwareMarginLoss(s=s, m1=m1, m2=m2, m3=m3, focal_gamma=f_gamma)
		self.text_center_loss = RepelContrastiveLoss(init_logit_scale=15, min_temp=1.0, max_temp=100.0)
		self.permuted_contrastive_loss = PermutedContrastiveLoss(margin=m4)
		# self.permuted_contrastive_loss = PermutedNTXentLoss(temperature=0.1)
		self.graph_embeddings = graph_embeddings
		self.graph_loss = GraphLoss(ignore_diagonal=True, min_graph_norm=1e-6, lambda_depth=0.0, depth_r_min=0.5, depth_r_max=2.0)      

	def cal_prompt_token(self, max_length=256):
		self.prompt_token_batches = self.text_encoder_wrapper.tokenize_prompts(self.label_prompts, max_length=max_length)

	def setup(self, stage=None):
		self.cal_prompt_token(max_length=256)
		self.text_encoder_wrapper.load_or_generate_prompt_cache(self.prompt_token_batches)

	def get_class_text_centers(self, labels=None, only_centers=False):
		cache = self.text_encoder_wrapper.prompt_layer_cache
		if labels is not None:
			labels = set(labels.tolist()) if not isinstance(labels, set) else labels
		else:
			labels = set(cache.keys())

		all_embeddings = []
		all_labels = []

		for class_id in sorted(labels):
			if class_id not in cache:
				continue

			hidden_states_10, attention_mask = cache[class_id]
			if self.training and hidden_states_10.size(0) > self.n_prompts:
				idx = torch.randperm(hidden_states_10.size(0), device=hidden_states_10.device)[:self.n_prompts]
				hidden_states_10 = hidden_states_10[idx]
				attention_mask = attention_mask[idx]

			hidden_states_10 = hidden_states_10.to(self.device)
			prompts_emb = self.text_encoder_wrapper.run_final_layer(hidden_states_10, attention_mask.to(self.device), pool_type=self.pool_type)

			all_embeddings.append(prompts_emb)
			all_labels.extend([class_id] * prompts_emb.size(0))

		if not all_embeddings:
			raise ValueError(f"No embeddings found for labels: {labels}")

		if only_centers:
			centers = [emb.mean(dim=0) for emb in all_embeddings]
			class_ids = [sorted(labels)[i] for i in range(len(all_embeddings))]
			return torch.tensor(class_ids, device=self.device), torch.stack(centers)
		
		else:
			label_tensor = torch.tensor(all_labels, device=self.device)
			embedding_tensor = torch.cat(all_embeddings, dim=0)
			return label_tensor, embedding_tensor
		


	def forward(self, x):
		emb = self.encoder(x)
		return emb
	

	def training_step(self, batch, batch_idx):
		
		expr_input, labels = batch
		if isinstance(expr_input, dict):
			expr_input = expr_input["X"]
		expr_embeddings = self(expr_input)
		expr_embeddings = F.normalize(expr_embeddings, dim=-1)

		prompt_labels, class_text_centers = self.get_class_text_centers(labels, only_centers=True)
		class_text_centers_normalized = F.normalize(class_text_centers, dim=-1)

		id_to_new_index = {k.item(): v for v, k in enumerate(prompt_labels)}
		mapped_labels = labels.clone()
		for old_id, new_id in id_to_new_index.items():
			mapped_labels[labels == old_id] = new_id

		# print(class_text_centers.shape, max(mapped_labels))
		loss_cell_text = self.loss_cell_text(expr_embeddings, class_text_centers_normalized, mapped_labels)
		loss_permuted, n_positive, pos_dis, neg_dis = self.permuted_contrastive_loss(expr_embeddings, mapped_labels)


		# prompt_labels, class_text_centers = self.get_class_text_centers(None, only_centers=True)
		# loss_cell_text = self.loss_cell_text(expr_embeddings, class_text_centers, labels)
		# loss_permuted, n_positive = self.permuted_contrastive_loss(expr_embeddings, labels)

		loss_text_text = self.text_center_loss(class_text_centers_normalized)

		if self.graph_embeddings is None:
			loss = self.l_ct * loss_cell_text + self.l_tt * loss_text_text + self.l_contrastive * loss_permuted
		else:
			graph_loss = self.graph_loss(class_text_centers, prompt_labels, self.graph_embeddings)
			loss = self.l_ct * loss_cell_text + self.l_tt * loss_text_text + self.l_contrastive * loss_permuted + self.l_graph * graph_loss



		
		## logging
		self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
		self.log("loss_cell_text", loss_cell_text, on_step=False, on_epoch=True, prog_bar=True)
		self.log("loss_text_text", loss_text_text, on_step=False, on_epoch=True, prog_bar=True)
		self.log("loss_permuted", loss_permuted, on_step=False, on_epoch=True, prog_bar=True)
		if self.graph_embeddings is not None:
			self.log("graph_loss", graph_loss, on_step=False, on_epoch=True, prog_bar=True)
		# self.log("positive ratio", n_positive/len(labels), on_step=True, on_epoch=True, prog_bar=True)
		# self.log("pos_dis", pos_dis.mean(), on_step=True, on_epoch=True, prog_bar=True)
		# self.log("neg_dis", neg_dis.mean(), on_step=True, on_epoch=True, prog_bar=True)

		# log the value of text center loss temperature
		return loss

	def configure_optimizers(self):
		optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.wd)
		return optimizer
	

def process_new_celltypes(model, cell_types, label_prompts = None, cache_path="./cache/cached_prompt_embeddings.pt"):
	## create the cache folder
	os.makedirs(os.path.dirname(cache_path), exist_ok=True)		
	int_to_ct = {i: ct for i, ct in enumerate(cell_types)}
	model.eval()

	if label_prompts is None:
		prompt_templates = [
			"A single-cell transcriptome from a {label} cell.",
			"This is the gene expression profile of a {label} cell.",
			"This cell is classified as a {label}.",
			"This cell type is called {label}.",
			"Semantic annotation for this cell: {label} cell.",
			"This is {label} cell.",
			"This is a cell of type {label}.",
			"A single-cell transcriptomic signature indicates a {label} cell.",
			"Gene expression pattern corresponds to a {label}.",
			"This single-cell profile matches a {label} cell."
			
			]
		cell_types = list(int_to_ct.values())
		label_prompts = {
			i: [template.format(label=label) for template in prompt_templates]
			for i, label in enumerate(cell_types)
		}
	else:
		## reorder the label_prompts to match the int_to_ct
		label_prompts = {i: label_prompts[ct] for i, ct in int_to_ct.items()}
		
	model.label_prompts = label_prompts

	model.prompt_token_batches = {
		class_id: model.text_encoder_wrapper.tokenizer(
			prompts,
			padding=True,
			truncation=True,
			max_length=256,  # or 512 depending on your prompt format
			return_tensors="pt"
		)
		for class_id, prompts in label_prompts.items()
	}
	model.on_train_start()
	model.text_encoder_wrapper.cache_path = cache_path
	if os.path.exists(model.text_encoder_wrapper.cache_path):
		os.remove(model.text_encoder_wrapper.cache_path)
	model.prompt_layer10_cache = {}
	model.setup()
