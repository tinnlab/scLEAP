import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from torch.nn import Parameter

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):

    def __init__(self, gamma=0, eps=1e-7, alpha = None):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.eps = eps
        self.ce = torch.nn.CrossEntropyLoss()

    def forward(self, input, target):
        logp = self.ce(input, target)
        p = torch.exp(-logp)
        loss = (1 - p) ** self.gamma * logp
        return loss.mean()


    


class LabelAwareMarginLoss(nn.Module):
    r"""
    Generalized margin loss:
        target logit = s * (cos(m1 * theta + m2) - m3)
        non-target   = s * cos(theta)

    Shapes:
        exprs: [N, D]  (sample features)
        text:  [C, D]  (class prototypes / text embeddings)
        label: [N]     (class indices in [0, C-1])
    """
    def __init__(
        self,
        s=15.0, m1=1.0, m2=0.30, m3=0.0,
        focal_gamma=2.0,
        easy_margin=False,
        eps=1e-6
    ):
        super().__init__()
        self.criterion = FocalLoss(gamma=focal_gamma)  # assumes logits input
        self.s = float(s)
        self.m1 = float(m1)
        self.m2 = float(m2)
        self.m3 = float(m3)
        self.easy_margin = easy_margin
        self.eps = eps

    def forward(self, exprs, text, label):
        # Cosine similarities (N x C); normalize for cosine space
        cosine = F.linear(F.normalize(exprs), F.normalize(text))
        cosine = cosine.clamp(-1.0 + self.eps, 1.0 - self.eps)

        # Angular transform for targets
        theta = torch.acos(cosine)                     # in [0, pi]
        phi = torch.cos(self.m1 * theta + self.m2)     # cos(m1*θ + m2)

        # Optional easy margin: fall back to cosθ when cosθ <= 0
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)

        # Apply CosFace-style margin only to targets later via one_hot
        phi = phi - self.m3

        # Mix target vs non-target logits
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        logits = (one_hot * phi + (1.0 - one_hot) * cosine) * self.s

        loss = self.criterion(logits, label)
        return loss.mean()


class RepelContrastiveLoss(nn.Module):
    def __init__(self, init_logit_scale=15.0, min_temp=1.0, max_temp=100.0):
        super().__init__()
        # Direct, learnable logit scale (no temperature, no log/exp)
        self.logit_scale = nn.Parameter(torch.tensor(float(init_logit_scale)))
        # Keeping the same names for drop-in compatibility
        self.min_temp = min_temp
        self.max_temp = max_temp

    def forward(self, embeddings):
        # Normalize vectors
        embeddings = F.normalize(embeddings, dim=-1)

        # Similarity matrix
        sim = embeddings @ embeddings.T

        # Clamp the direct scale
        scale = self.logit_scale.clamp(self.min_temp, self.max_temp)

        # Scale similarities
        logits = sim * scale

        # Standard within-batch labels
        targets = torch.arange(logits.size(0), device=logits.device)

        loss = F.cross_entropy(logits, targets)
        return loss


### new loss

class ContrastiveLoss(nn.Module):
    def __init__(self, margin=2.0):
        super().__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        # label: 0 -> similar, 1 -> dissimilar
        euclidean_distance = F.pairwise_distance(output1, output2)
        pos = (1 - label) * euclidean_distance.pow(2)
        neg = label * torch.clamp(self.margin - euclidean_distance, min=0.0).pow(2)
        loss_contrastive = torch.mean(pos + neg)
        return loss_contrastive, pos, neg


class PermutedContrastiveLoss(nn.Module):
    """
    Takes a batch of embeddings and class labels, creates random pairs
    via permutation, builds binary labels, and applies ContrastiveLoss.
    """

    def __init__(self, margin=2.0, avoid_self_pairs=True):
        super().__init__()
        self.base_loss = ContrastiveLoss(margin=margin)
        self.avoid_self_pairs = avoid_self_pairs

    def forward(self, embeddings, targets):
        """
        embeddings: Tensor of shape [B, D]
        targets:    Tensor of shape [B] or [B, 1] with class indices

        Returns:
            scalar contrastive loss
        """
        # Ensure 1D targets
        if targets.dim() > 1:
            targets = targets.view(-1)

        B = embeddings.size(0)
        device = embeddings.device

        if self.avoid_self_pairs and B > 1:
            # Deterministic "no self-pair" permutation via rotation
            perm = torch.roll(torch.arange(B, device=device), shifts=1)
        else:
            # Fully random permutation (may include self-pairs)
            perm = torch.randperm(B, device=device)

        emb1 = embeddings
        emb2 = embeddings[perm]

        t1 = targets
        t2 = targets[perm]

        # Binary labels for contrastive loss: 0 = same, 1 = different
        pair_labels = (t1 != t2).float()

        # Make sure labels are on same device/dtype as embeddings
        pair_labels = pair_labels.to(embeddings.dtype)

        loss, pos_dis, neg_dis = self.base_loss(emb1, emb2, pair_labels)
        n_positive = (pair_labels == 0).float().sum()

        return loss, n_positive, pos_dis, neg_dis




import torch
import torch.nn as nn
import torch.nn.functional as F


def project_to_poincare_ball(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    norm = x.norm(dim=-1, keepdim=True)
    max_norm = 1.0 - eps
    scale = torch.where(norm > max_norm, max_norm / norm, torch.ones_like(norm))
    return x * scale


def pairwise_poincare_distance(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """
    x: [N, d] in Poincaré ball
    returns: [N, N] distances
    """
    x = project_to_poincare_ball(x, eps=eps)
    x2 = (x * x).sum(dim=-1, keepdim=True)         # [N, 1]
    sqdist = x2 + x2.t() - 2.0 * x @ x.t()         # [N, N]
    denom = (1.0 - x2) * (1.0 - x2.t())
    denom = denom.clamp_min(eps)
    z = 1.0 + 2.0 * sqdist / denom
    z = z.clamp_min(1.0 + eps)
    return torch.acosh(z)




# class GraphLoss(nn.Module):
#     """
#     Simple graph–text alignment loss.

#     1) Distance matching (main term)
#        - Graph: Poincaré distance d_graph(i,j)
#        - Text:  cosine distance d_text(i,j) = 1 - cos(t_i, t_j)
#        Loss: MSE between d_text and (alpha_graph * d_graph), optionally
#        ignoring the diagonal.

#     2) Optional depth / norm matching (regularizer)
#        - Graph depth: Poincaré radius r_graph = ||project_to_poincare_ball(g_i)|| in [0, 1)
#        - Target text radius: r_target = depth_r_min + (depth_r_max - depth_r_min) * r_graph
#        - Text radius: r_text = ||t_i|| (raw norm, not normalized)
#        Loss: MSE(r_text, r_target) weighted by lambda_depth.
#     """

#     def __init__(
#         self,
#         alpha_graph: float = 1.0,      # scalar to rescale graph distances
#         ignore_diagonal: bool = True,
#         min_graph_norm: float = 1e-6,
#         lambda_depth: float = 0.0,     # weight of depth / norm-matching term
#         depth_r_min: float = 0.5,      # min desired text norm (for shallow nodes)
#         depth_r_max: float = 2.0,      # max desired text norm (for deep nodes)
#     ):
#         super().__init__()
#         self.alpha_graph = alpha_graph
#         self.ignore_diagonal = ignore_diagonal
#         self.min_graph_norm = min_graph_norm
#         self.lambda_depth = lambda_depth
#         self.depth_r_min = depth_r_min
#         self.depth_r_max = depth_r_max

#     def forward(
#         self,
#         class_text_centers: torch.Tensor,  # [B, d_text] (RAW, not normalized)
#         prompt_labels: torch.Tensor,       # [B]
#         graph_embeddings,                  # [num_classes, d_graph] Poincaré
#     ):
#         if graph_embeddings is None:
#             raise ValueError("GraphLoss: graph_embeddings cannot be None.")

#         device = class_text_centers.device

#         # ensure tensor
#         if not torch.is_tensor(graph_embeddings):
#             graph_embeddings = torch.as_tensor(
#                 graph_embeddings, dtype=class_text_centers.dtype
#             )
#         graph_embeddings = graph_embeddings.to(device)  # [C, d_graph]

#         # select batch graph embeddings
#         graph_batch = graph_embeddings[prompt_labels]   # [B, d_graph]
#         B = class_text_centers.size(0)
#         if B < 2:
#             return class_text_centers.new_tensor(0.0)

#         # filter invalid graph vectors
#         graph_norms = graph_batch.norm(dim=-1)          # [B]
#         valid_mask = graph_norms > self.min_graph_norm
#         if valid_mask.sum() < 2:
#             return class_text_centers.new_tensor(0.0)
        
#         text_valid = class_text_centers[valid_mask]     # [Bv, d_text] (raw)
#         graph_valid = graph_batch[valid_mask]           # [Bv, d_graph]
#         Bv = text_valid.size(0)
#         device = text_valid.device

#         # ==================== 1) DISTANCE MATCHING ====================

#         # ----- GRAPH DISTANCES (Poincaré) -----
#         # pairwise_poincare_distance: [Bv, Bv]
#         dist_graph = pairwise_poincare_distance(graph_valid)       # [Bv, Bv]
#         dist_graph = self.alpha_graph * dist_graph                 # optional rescale

#         # ----- TEXT DISTANCES (COSINE) -----
#         text_normed = F.normalize(text_valid, dim=-1)              # [Bv, d_text]
#         cos = text_normed @ text_normed.t()                        # [Bv, Bv], in [-1, 1]
#         dist_text = 1.0 - cos                                      # [Bv, Bv], in [0, 2]

#         # optionally ignore diagonal (self-distances)
#         if self.ignore_diagonal:
#             mask = ~torch.eye(Bv, dtype=torch.bool, device=device)
#             dg = dist_graph[mask]
#             dt = dist_text[mask]
#         else:
#             dg = dist_graph.flatten()
#             dt = dist_text.flatten()

#         # distance matching loss: text cosine-distances ≈ scaled graph distances
#         dist_loss = F.mse_loss(dt, dg)

#         total_loss = dist_loss

#         # ==================== 2) DEPTH / NORM MATCHING ====================

#         if self.lambda_depth > 0.0:
#             # Poincaré radius in [0,1)
#             r_graph = project_to_poincare_ball(graph_valid).norm(dim=-1)  # [Bv]

#             # map to desired text radius range
#             r_target = self.depth_r_min + (self.depth_r_max - self.depth_r_min) * r_graph

#             # text norms (NOTE: we use raw text_valid, not normalized)
#             r_text = text_valid.norm(dim=-1)   # [Bv]

#             depth_loss = F.mse_loss(r_text, r_target)
#             total_loss = total_loss + self.lambda_depth * depth_loss

#         return total_loss




class GraphLoss(nn.Module):
    """
    Graph–text distance alignment loss with normalized distances.

    1) Distance matching (main term)
       - Graph: Poincaré distance d_graph(i,j)
       - Text:  cosine distance d_text(i,j) = 1 - cos(t_i, t_j)
       - Both distance sets are normalized before matching:
            d_graph_norm = d_graph / mean(d_graph)
            d_text_norm  = d_text  / mean(d_text)
       Loss: MSE(d_text_norm, d_graph_norm), optionally ignoring diagonal.

    2) Optional depth / norm matching (regularizer)
       - Graph depth: Poincaré radius r_graph = ||project_to_poincare_ball(g_i)|| in [0, 1)
       - Target text radius: r_target = depth_r_min + (depth_r_max - depth_r_min) * r_graph
       - Text radius: r_text = ||t_i|| (raw norm, not normalized)
       Loss: MSE(r_text, r_target) weighted by lambda_depth.
    """

    def __init__(
        self,
        ignore_diagonal: bool = True,
        min_graph_norm: float = 1e-6,
        lambda_depth: float = 0.0,
        depth_r_min: float = 0.5,
        depth_r_max: float = 2.0,
        eps: float = 1e-8,
    ):
        super().__init__()
        self.ignore_diagonal = ignore_diagonal
        self.min_graph_norm = min_graph_norm
        self.lambda_depth = lambda_depth
        self.depth_r_min = depth_r_min
        self.depth_r_max = depth_r_max
        self.eps = eps

    def forward(
        self,
        class_text_centers: torch.Tensor,  # [B, d_text] (RAW, not normalized)
        prompt_labels: torch.Tensor,       # [B]
        graph_embeddings,                  # [num_classes, d_graph] Poincaré
    ):
        if graph_embeddings is None:
            raise ValueError("GraphLoss: graph_embeddings cannot be None.")

        device = class_text_centers.device
        dtype = class_text_centers.dtype

        # ensure tensor
        if not torch.is_tensor(graph_embeddings):
            graph_embeddings = torch.as_tensor(graph_embeddings, dtype=dtype)
        graph_embeddings = graph_embeddings.to(device)

        # select batch graph embeddings
        graph_batch = graph_embeddings[prompt_labels]   # [B, d_graph]
        B = class_text_centers.size(0)
        if B < 2:
            return class_text_centers.new_tensor(0.0)

        # filter invalid graph vectors
        graph_norms = graph_batch.norm(dim=-1)          # [B]
        valid_mask = graph_norms > self.min_graph_norm
        if valid_mask.sum() < 2:
            return class_text_centers.new_tensor(0.0)

        text_valid = class_text_centers[valid_mask]     # [Bv, d_text]
        graph_valid = graph_batch[valid_mask]           # [Bv, d_graph]
        Bv = text_valid.size(0)
        device = text_valid.device

        # ==================== 1) DISTANCE MATCHING ====================

        # ----- GRAPH DISTANCES (Poincaré) -----
        dist_graph = pairwise_poincare_distance(graph_valid)   # [Bv, Bv]

        # ----- TEXT DISTANCES (Cosine) -----
        text_normed = F.normalize(text_valid, dim=-1)          # [Bv, d_text]
        cos = text_normed @ text_normed.t()                    # [Bv, Bv]
        cos = cos.clamp(-1.0, 1.0)
        dist_text = 1.0 - cos                                  # [Bv, Bv]

        # select entries
        if self.ignore_diagonal:
            mask = ~torch.eye(Bv, dtype=torch.bool, device=device)
            dg = dist_graph[mask]
            dt = dist_text[mask]
        else:
            dg = dist_graph.reshape(-1)
            dt = dist_text.reshape(-1)

        # if no valid pairs remain
        if dg.numel() == 0:
            return class_text_centers.new_tensor(0.0)

        # normalize distances to comparable scale
        dg = dg / (dg.mean().detach() + self.eps)
        dt = dt / (dt.mean().detach() + self.eps)

        dist_loss = F.mse_loss(dt, dg)

        total_loss = dist_loss

        # ==================== 2) DEPTH / NORM MATCHING ====================

        if self.lambda_depth > 0.0:
            # Poincaré radius in [0, 1)
            r_graph = project_to_poincare_ball(graph_valid).norm(dim=-1)  # [Bv]

            # map to desired text radius range
            r_target = self.depth_r_min + (
                self.depth_r_max - self.depth_r_min
            ) * r_graph

            # raw text norm
            r_text = text_valid.norm(dim=-1)

            depth_loss = F.mse_loss(r_text, r_target)
            total_loss = total_loss + self.lambda_depth * depth_loss

        return total_loss