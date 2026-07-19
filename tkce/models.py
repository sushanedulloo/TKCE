"""Neural architectures: encoder phi, downstream heads, and the joint model.

  * SiameseEncoder - the shared-weight MLP phi that maps a row to an embedding.
                     L2-normalized so <phi(x_i), phi(x_j)> is a cosine in [-1,1].
  * MLPHead        - a plain MLP classifier/regressor (used as the downstream
                     head on embeddings, and as the raw-feature MLP baseline).
  * TabResNet      - the tabular ResNet of Gorishniy et al. 2021 ("Revisiting
                     Deep Learning Models for Tabular Data"): input projection +
                     N pre-norm residual blocks + prediction head. Used both on
                     raw features (baseline) and on embeddings (our method).
  * JointModel     - encoder + head in one module, for end-to-end training with
                     an auxiliary contrastive loss (the "train the embedding
                     layer jointly" regime).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SiameseEncoder(nn.Module):
    def __init__(self, in_dim, hidden_dims=(256, 128), embedding_dim=64,
                 dropout=0.1, batchnorm=True, l2_normalize=True):
        super().__init__()
        self.l2_normalize = l2_normalize
        self.embedding_dim = embedding_dim
        layers, prev = [], in_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            if batchnorm:
                layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, embedding_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        z = self.net(x)
        return F.normalize(z, p=2, dim=1) if self.l2_normalize else z


class MLPHead(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dims=(256, 128),
                 dropout=0.1, batchnorm=True):
        super().__init__()
        layers, prev = [], in_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            if batchnorm:
                layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class _ResBlock(nn.Module):
    def __init__(self, d, d_hidden, dropout):
        super().__init__()
        self.norm = nn.BatchNorm1d(d)
        self.lin1 = nn.Linear(d, d_hidden)
        self.lin2 = nn.Linear(d_hidden, d)
        self.drop1 = nn.Dropout(dropout)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x):
        z = self.norm(x)
        z = F.relu(self.lin1(z))
        z = self.drop1(z)
        z = self.lin2(z)
        z = self.drop2(z)
        return x + z


class TabResNet(nn.Module):
    """Gorishniy et al. 2021 tabular ResNet."""

    def __init__(self, in_dim, out_dim, d=192, d_hidden=256,
                 n_blocks=3, dropout=0.1):
        super().__init__()
        self.proj = nn.Linear(in_dim, d)
        self.blocks = nn.ModuleList(
            [_ResBlock(d, d_hidden, dropout) for _ in range(n_blocks)])
        self.head_norm = nn.BatchNorm1d(d)
        self.head = nn.Linear(d, out_dim)

    def forward(self, x):
        x = self.proj(x)
        for b in self.blocks:
            x = b(x)
        x = F.relu(self.head_norm(x))
        return self.head(x)


class FTTransformer(nn.Module):
    """Feature-Tokenizer Transformer (Gorishniy et al. 2021).

    Each numerical feature is tokenized (x_j * w_j + b_j) and each categorical
    feature is embedded; a [CLS] token is prepended and a Transformer encoder
    mixes them. The CLS representation feeds the prediction head. Expects the
    numerical-encoded matrix (numerical columns first, then ordinal categoricals).
    """

    def __init__(self, n_num, cat_cardinalities, out_dim, d_token=64,
                 n_blocks=3, n_heads=8, dropout=0.1):
        super().__init__()
        self.n_num = n_num
        if n_num > 0:
            self.num_w = nn.Parameter(torch.randn(n_num, d_token) * 0.02)
            self.num_b = nn.Parameter(torch.zeros(n_num, d_token))
        self.cat_embs = nn.ModuleList(
            [nn.Embedding(int(c) + 2, d_token) for c in cat_cardinalities])
        self.cls = nn.Parameter(torch.randn(1, 1, d_token) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d_token, nhead=n_heads, dim_feedforward=d_token * 2,
            dropout=dropout, activation="gelu", batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_blocks)
        self.norm = nn.LayerNorm(d_token)
        self.head = nn.Linear(d_token, out_dim)

    def forward(self, x):
        B = x.shape[0]
        toks = []
        if self.n_num > 0:
            xn = x[:, :self.n_num]                       # (B, n_num)
            toks.append(xn.unsqueeze(-1) * self.num_w + self.num_b)  # (B,n_num,d)
        if len(self.cat_embs) > 0:
            xc = (x[:, self.n_num:].long() + 1)          # shift unknown(-1) -> 0
            cat = []
            for j, emb in enumerate(self.cat_embs):
                idx = xc[:, j].clamp(0, emb.num_embeddings - 1)
                cat.append(emb(idx))
            toks.append(torch.stack(cat, dim=1))         # (B, n_cat, d)
        tok = torch.cat(toks, dim=1) if toks else \
            torch.zeros(B, 0, self.cls.shape[-1], device=x.device)
        tok = torch.cat([self.cls.expand(B, -1, -1), tok], dim=1)
        z = self.encoder(tok)
        return self.head(self.norm(z[:, 0]))


class _PeriodicEmbedding(nn.Module):
    """Periodic numerical embedding (Gorishniy et al. 2022, arXiv 2203.05556)."""

    def __init__(self, n_num, k=16, sigma=1.0):
        super().__init__()
        self.coeffs = nn.Parameter(torch.randn(n_num, k) * sigma)

    def forward(self, xn):                               # (B, n_num)
        v = 2 * math.pi * xn.unsqueeze(-1) * self.coeffs  # (B, n_num, k)
        return torch.cat([torch.sin(v), torch.cos(v)], dim=-1)  # (B, n_num, 2k)


class NumEmbedMLP(nn.Module):
    """MLP with periodic numerical embeddings + categorical embeddings (MLP-PLR).

    A direct embedding-based rival to TKCE: numerical features get periodic
    (sin/cos) embeddings, categoricals get learned embeddings, then an MLP.
    """

    def __init__(self, n_num, cat_cardinalities, out_dim, k=16, d_cat=8,
                 hidden_dims=(256, 128), dropout=0.1, sigma=1.0):
        super().__init__()
        self.n_num = n_num
        in_dim = 0
        if n_num > 0:
            self.periodic = _PeriodicEmbedding(n_num, k, sigma)
            self.num_lin = nn.Linear(n_num * 2 * k, n_num * 2 * k)
            in_dim += n_num * 2 * k
        self.cat_embs = nn.ModuleList(
            [nn.Embedding(int(c) + 2, d_cat) for c in cat_cardinalities])
        in_dim += len(cat_cardinalities) * d_cat
        layers, prev = [], max(in_dim, 1)
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(),
                       nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, out_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        parts = []
        if self.n_num > 0:
            xn = x[:, :self.n_num]
            e = self.periodic(xn).flatten(1)             # (B, n_num*2k)
            parts.append(F.relu(self.num_lin(e)))
        if len(self.cat_embs) > 0:
            xc = (x[:, self.n_num:].long() + 1)
            for j, emb in enumerate(self.cat_embs):
                parts.append(emb(xc[:, j].clamp(0, emb.num_embeddings - 1)))
        z = torch.cat(parts, dim=1) if parts else x
        return self.mlp(z)


class JointModel(nn.Module):
    """Encoder phi + head g, exposing both the embedding and the prediction."""

    def __init__(self, encoder: SiameseEncoder, head: nn.Module):
        super().__init__()
        self.encoder = encoder
        self.head = head

    def forward(self, x):
        z = self.encoder(x)
        return self.head(z), z


def head_out_dim(dataset) -> int:
    return dataset.n_classes if dataset.task_type == "classification" else 1


def build_head(kind: str, in_dim: int, out_dim: int, cfg: dict) -> nn.Module:
    if kind == "mlp":
        return MLPHead(in_dim, out_dim,
                       hidden_dims=tuple(cfg.get("hidden_dims", (256, 128))),
                       dropout=cfg.get("dropout", 0.1),
                       batchnorm=cfg.get("batchnorm", True))
    if kind == "tabresnet":
        return TabResNet(in_dim, out_dim,
                         d=cfg.get("d", 192), d_hidden=cfg.get("d_hidden", 256),
                         n_blocks=cfg.get("n_blocks", 3),
                         dropout=cfg.get("dropout", 0.1))
    if kind == "ft_transformer":
        return FTTransformer(cfg["n_num"], cfg.get("cat_cardinalities", []), out_dim,
                             d_token=cfg.get("d_token", 64),
                             n_blocks=cfg.get("n_blocks", 3),
                             n_heads=cfg.get("n_heads", 8),
                             dropout=cfg.get("dropout", 0.1))
    if kind == "num_embed_mlp":
        return NumEmbedMLP(cfg["n_num"], cfg.get("cat_cardinalities", []), out_dim,
                           k=cfg.get("k_freq", 16), d_cat=cfg.get("d_cat", 8),
                           hidden_dims=tuple(cfg.get("hidden_dims", (256, 128))),
                           dropout=cfg.get("dropout", 0.1))
    raise ValueError(f"unknown head kind '{kind}'")
