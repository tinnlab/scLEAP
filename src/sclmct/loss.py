import torch
import torch.nn as nn
import math
import torch.nn.functional as F

# CLIPLoss, CellTextContrastiveLoss, RepelContrastiveLoss
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




class CellTextContrastiveLoss(nn.Module):
    r"""Implement of large margin arc distance: :
        Args:
            in_features: size of each input sample
            out_features: size of each output sample
            s: norm of input feature
            m: margin

            cos(theta + m)
        """
    def __init__(self, s=15, m=0.30, focal_gamma = 2, easy_margin=False):
        super(CellTextContrastiveLoss, self).__init__()

        self.criterion = FocalLoss(gamma=focal_gamma)
        self.s = s
        self.m = m

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, exprs, text, label):
        # --------------------------- cos(theta) & phi(theta) ---------------------------
        cosine = F.linear(F.normalize(exprs), F.normalize(text))
        sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        # --------------------------- convert label to one-hot ---------------------------
        # one_hot = torch.zeros(cosine.size(), requires_grad=True, device='cuda')
        one_hot = torch.zeros(cosine.size(), device='cuda')
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)  # you can use torch.where if your torch.__version__ is 0.4
        output *= self.s
        # print(output)
        loss = self.criterion(output, label)
        return loss.mean()

class TextCentersContrastiveLoss(nn.Module):
    def __init__(self, init_temperature=0.07, min_temp=1.0, max_temp=100.0):
        super().__init__()
        self.logit_scale = nn.Parameter(torch.tensor(1 / init_temperature).log())
        self.min_temp = min_temp
        self.max_temp = max_temp

    def forward(self, embeddings):
        # Normalize vectors
        embeddings = F.normalize(embeddings, dim=-1)
        sim = embeddings @ embeddings.T
        scale = self.logit_scale.exp().clamp(self.min_temp, self.max_temp)
        
        logits = embeddings @ embeddings.T * scale

        targets = torch.arange(logits.size(0), device=logits.device)
        loss = F.cross_entropy(logits, targets)
        return loss

