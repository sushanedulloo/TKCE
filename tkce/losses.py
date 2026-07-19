"""Contrastive / metric-learning objectives for training phi against the kernel.

Seven losses, spanning the standard contrastive-learning families:

  infonce           SimCLR/CPC NT-Xent: one positive vs in-batch negatives, temp.
  kernel_regression MSE between <phi(x_i),phi(x_j)> and the kernel value K.
  contrastive       Hadsell/Chopra margin pair loss (pull positives, push negs).
  triplet           FaceNet: anchor closer to positive than negative by a margin.
  supcon            Supervised Contrastive: multiple positives per anchor.
  aninfonce         Anisotropic InfoNCE (arXiv 2407.00143): learnable per-dim scale.
  clip_infonce      CLIP objective: symmetric InfoNCE over the anchor x positive
                    matrix with a LEARNABLE temperature (single-modality adaption).

`build_contrastive` returns (loss_module, data_regime). The data regime selects
how batches are drawn:
  "regression" -> (x_i, x_j, K) soft pairs        (kernel_regression)
  "supcon"     -> grouped anchor+positives batches (supcon)
  "anchorpos"  -> (anchor, positive) pairs; negatives are built in-batch. All of
                  infonce/aninfonce/clip_infonce/contrastive/triplet share this
                  path and are dispatched by `apply_pair_loss`.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# 1. InfoNCE (SimCLR NT-Xent, in-batch negatives)
# --------------------------------------------------------------------------- #
class InfoNCELoss(nn.Module):
    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.tau = temperature

    def forward(self, zi, zj):
        B = zi.shape[0]
        z = torch.cat([zi, zj], dim=0)                 # (2B, d)
        sim = z @ z.t() / self.tau
        eye = torch.eye(2 * B, device=z.device, dtype=torch.bool)
        sim.masked_fill_(eye, float("-inf"))
        targets = (torch.arange(2 * B, device=z.device) + B) % (2 * B)
        return F.cross_entropy(sim, targets)


# --------------------------------------------------------------------------- #
# 2. Kernel regression (CatFormer objective)
# --------------------------------------------------------------------------- #
class KernelRegressionLoss(nn.Module):
    def forward(self, zi, zj, K):
        return F.mse_loss((zi * zj).sum(dim=1), K)


# --------------------------------------------------------------------------- #
# 3. Contrastive pair loss (Hadsell/Chopra/LeCun 2006)
# --------------------------------------------------------------------------- #
class ContrastiveLoss(nn.Module):
    def __init__(self, margin: float = 1.0):
        super().__init__()
        self.margin = margin

    def forward(self, zi, zj, similar):
        d = F.pairwise_distance(zi, zj)
        pos = similar * d.pow(2)
        neg = (1 - similar) * F.relu(self.margin - d).pow(2)
        return (pos + neg).mean()


# --------------------------------------------------------------------------- #
# 4. Triplet loss (FaceNet, Schroff 2015)
# --------------------------------------------------------------------------- #
class TripletLoss(nn.Module):
    def __init__(self, margin: float = 1.0):
        super().__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        dp = F.pairwise_distance(anchor, positive)
        dn = F.pairwise_distance(anchor, negative)
        return F.relu(dp.pow(2) - dn.pow(2) + self.margin).mean()


# --------------------------------------------------------------------------- #
# 5. Supervised Contrastive (Khosla 2020) — multiple positives per anchor
# --------------------------------------------------------------------------- #
class SupConLoss(nn.Module):
    """z: (N, d) L2-normed; pos_mask: (N, N) bool with True for positive pairs
    (diagonal excluded). Averages the InfoNCE log-prob over each anchor's
    positives, then over anchors that have at least one positive."""

    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.tau = temperature

    def forward(self, z, pos_mask):
        N = z.shape[0]
        sim = z @ z.t() / self.tau
        self_mask = ~torch.eye(N, device=z.device, dtype=torch.bool)
        sim = sim.masked_fill(~self_mask, float("-inf"))
        logits = sim - sim.max(dim=1, keepdim=True).values.detach()
        exp = torch.exp(logits) * self_mask
        log_prob = logits - torch.log(exp.sum(dim=1, keepdim=True) + 1e-12)
        pos = pos_mask & self_mask
        pos_count = pos.sum(dim=1)
        mean_log_prob = (pos * log_prob).sum(dim=1) / pos_count.clamp(min=1)
        valid = pos_count > 0
        if valid.sum() == 0:
            return sim.new_zeros(())
        return -mean_log_prob[valid].mean()


# --------------------------------------------------------------------------- #
# 6. AnInfoNCE — anisotropic InfoNCE (arXiv 2407.00143)
# --------------------------------------------------------------------------- #
class AnInfoNCELoss(nn.Module):
    """InfoNCE with a learnable positive per-dimension scale s_d = softplus(theta_d),
    so a few dominant factors are stretched instead of collapsing the rest."""

    def __init__(self, dim: int, temperature: float = 0.1):
        super().__init__()
        self.tau = temperature
        self.log_scale = nn.Parameter(torch.zeros(dim))

    def forward(self, zi, zj):
        B = zi.shape[0]
        z = torch.cat([zi, zj], dim=0)
        s = F.softplus(self.log_scale)
        sim = (z * s) @ z.t() / self.tau
        eye = torch.eye(2 * B, device=z.device, dtype=torch.bool)
        sim.masked_fill_(eye, float("-inf"))
        targets = (torch.arange(2 * B, device=z.device) + B) % (2 * B)
        return F.cross_entropy(sim, targets)


# --------------------------------------------------------------------------- #
# 7. CLIP-style symmetric InfoNCE with a learnable temperature (arXiv 2103.00020)
# --------------------------------------------------------------------------- #
class CLIPInfoNCELoss(nn.Module):
    """Anchors and their positives are the two 'views'. Builds the B x B cosine
    matrix, treats the diagonal as positives, and averages the anchor->positive
    and positive->anchor cross-entropies. Temperature is learned (logit scale),
    as in CLIP."""

    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.log_temp = nn.Parameter(torch.tensor(math.log(1.0 / temperature)))

    def forward(self, zi, zj):
        B = zi.shape[0]
        scale = self.log_temp.exp().clamp(max=100.0)
        logits = scale * zi @ zj.t()                   # (B, B)
        targets = torch.arange(B, device=zi.device)
        return 0.5 * (F.cross_entropy(logits, targets)
                      + F.cross_entropy(logits.t(), targets))


# --------------------------------------------------------------------------- #
# Dispatch
# --------------------------------------------------------------------------- #
_ANCHORPOS = {"infonce", "aninfonce", "clip_infonce", "contrastive", "triplet"}

ALL_LOSSES = ["infonce", "kernel_regression", "contrastive", "triplet",
              "supcon", "aninfonce", "clip_infonce"]


def build_contrastive(name, temperature=0.1, dim=None, margin=1.0):
    """Return (loss_module, data_regime)."""
    if name == "infonce":
        return InfoNCELoss(temperature), "anchorpos"
    if name == "kernel_regression":
        return KernelRegressionLoss(), "regression"
    if name == "contrastive":
        return ContrastiveLoss(margin), "anchorpos"
    if name == "triplet":
        return TripletLoss(margin), "anchorpos"
    if name == "supcon":
        return SupConLoss(temperature), "supcon"
    if name == "aninfonce":
        return AnInfoNCELoss(dim or 64, temperature), "anchorpos"
    if name == "clip_infonce":
        return CLIPInfoNCELoss(temperature), "anchorpos"
    raise ValueError(f"unknown contrastive loss '{name}'")


def apply_pair_loss(name, criterion, zi, zj):
    """Compute an anchorpos-family loss from encoded (anchor, positive) batches.
    Negatives are constructed in-batch (2B stacking, or a within-batch shuffle)."""
    if name in ("infonce", "aninfonce", "clip_infonce"):
        return criterion(zi, zj)
    perm = torch.randperm(zj.shape[0], device=zj.device)
    if name == "triplet":
        return criterion(zi, zj, zj[perm])
    if name == "contrastive":
        z_a = torch.cat([zi, zi], dim=0)
        z_b = torch.cat([zj, zj[perm]], dim=0)
        flag = torch.cat([torch.ones(zi.shape[0], device=zi.device),
                          torch.zeros(zi.shape[0], device=zi.device)])
        return criterion(z_a, z_b, flag)
    raise ValueError(f"loss '{name}' is not an anchorpos-family loss")
