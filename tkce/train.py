"""Training routines for the three neural regimes.

  pretrain_encoder  : Stage A of the two-stage method. Train phi contrastively
                      against the tree kernel (no labels used here).
  train_head        : Stage B. Train a head on features (raw or embeddings) with
                      the supervised task loss. Used for baselines AND for the
                      two-stage downstream head (with a frozen/finetuned encoder).
  train_joint       : End-to-end regime. One net = encoder + head; loss is
                      task_loss + lambda * contrastive_loss, optimized together.

Device is auto-selected (cuda > mps > cpu) but can be forced.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from .losses import apply_pair_loss, build_contrastive
from .models import (JointModel, SiameseEncoder, build_head, head_out_dim)
from .pairs import (AnchorPositiveDataset, SampledPositiveIndex, SoftPairDataset)


def resolve_device(pref="auto"):
    if pref not in ("auto", None):
        return torch.device(pref)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _task_loss(task_type):
    return (lambda out, y: F.cross_entropy(out, y)) if task_type == "classification" \
        else (lambda out, y: F.mse_loss(out.squeeze(-1), y))


def _cycle(loader):
    """Infinite iterator over a DataLoader (never raises StopIteration)."""
    while True:
        for b in loader:
            yield b


# --------------------------------------------------------------------------- #
# Stage A: contrastive pretraining of phi
# --------------------------------------------------------------------------- #
def pretrain_encoder(X_train, kernel, cfg, device=None):
    device = device or resolve_device(cfg.get("device", "auto"))
    torch.manual_seed(cfg.get("seed", 0))

    # Cap the rows used for pair mining / pretraining on very large datasets.
    # The encoder still applies to every row at encode time.
    max_rows = cfg.get("max_pretrain_rows", 10_000)
    Xp = X_train
    if len(X_train) > max_rows:
        sub = np.random.default_rng(cfg.get("seed", 0)).choice(
            len(X_train), max_rows, replace=False)
        Xp = np.ascontiguousarray(X_train[sub])

    enc = SiameseEncoder(
        Xp.shape[1],
        hidden_dims=tuple(cfg.get("enc_hidden", (256, 128))),
        embedding_dim=cfg.get("embedding_dim", 64),
        dropout=cfg.get("enc_dropout", 0.1),
    ).to(device)

    loss_name = cfg.get("contrastive_loss", "infonce")
    criterion, regime = build_contrastive(
        loss_name, cfg.get("temperature", 0.1),
        dim=cfg.get("embedding_dim", 64), margin=cfg.get("margin", 1.0))
    # Some losses (AnInfoNCE, CLIP) carry learnable parameters -> optimize them too.
    params = list(enc.parameters()) + list(criterion.to(device).parameters())
    opt = torch.optim.AdamW(params, lr=cfg.get("enc_lr", 1e-3),
                            weight_decay=cfg.get("weight_decay", 1e-4))
    leaves = kernel.leaves(Xp)
    epochs = cfg.get("pretrain_epochs", 30)

    # ---- Regime: SupCon (grouped anchor + multiple positives) ----
    if regime == "supcon":
        return _pretrain_supcon(enc, criterion, opt, Xp, kernel, leaves, cfg,
                                device, epochs)

    # ---- Regime: kernel regression (soft K targets) ----
    if regime == "regression":
        ds = SoftPairDataset(Xp, kernel, leaves,
                             n_pairs=max(20000, 20 * len(Xp)),
                             seed=cfg.get("seed", 0))
    # ---- Regime: anchor/positive family (in-batch negatives) ----
    else:
        idx = SampledPositiveIndex(kernel, leaves,
                                   pos_threshold=cfg.get("pos_threshold", 0.6),
                                   max_pos=cfg.get("max_pos", 50),
                                   seed=cfg.get("seed", 0))
        ds = AnchorPositiveDataset(Xp, idx, seed=cfg.get("seed", 0))
        if len(ds) == 0:
            # A heavy perturbation can wipe out all positives; don't crash the
            # sweep — return the (untrained) encoder so the caller still runs.
            import warnings
            warnings.warn("No positive pairs mined; encoder left untrained.")
            return enc, [float("nan")]
    loader = DataLoader(ds, batch_size=cfg.get("batch_size", 256),
                        shuffle=True, drop_last=False)

    history = []
    for _ in range(epochs):
        enc.train(); running = 0.0; nb = 0
        for batch in loader:
            opt.zero_grad()
            if regime == "regression":
                xi, xj, K = (b.to(device) for b in batch)
                loss = criterion(enc(xi), enc(xj), K)
            else:
                xi, xj = (b.to(device) for b in batch)
                loss = apply_pair_loss(loss_name, criterion, enc(xi), enc(xj))
            loss.backward(); opt.step()
            running += loss.item(); nb += 1
        history.append(running / max(nb, 1))
    return enc, history


def _pretrain_supcon(enc, criterion, opt, Xp, kernel, leaves, cfg, device, epochs):
    """SupCon: each batch is several anchors, each with a few of its positives;
    same-anchor rows are positives of one another (a within-batch group mask)."""
    idx = SampledPositiveIndex(kernel, leaves,
                               pos_threshold=cfg.get("pos_threshold", 0.6),
                               max_pos=cfg.get("max_pos", 50),
                               seed=cfg.get("seed", 0))
    valid = [i for i in range(idx.n) if idx.has_positive(i)]
    if not valid:
        import warnings
        warnings.warn("No positive pairs mined; encoder left untrained.")
        return enc, [float("nan")]
    Xt = torch.from_numpy(np.ascontiguousarray(Xp)).to(device)
    rng = np.random.default_rng(cfg.get("seed", 0))
    G = cfg.get("supcon_groups", 32)          # anchors per batch
    P = cfg.get("supcon_positives", 4)        # positives per anchor
    history = []
    for _ in range(epochs):
        enc.train(); running = 0.0; nb = 0
        order = rng.permutation(valid)
        for s in range(0, len(order), G):
            chunk = order[s:s + G]
            rows, groups = [], []
            for gi, a in enumerate(chunk):
                pos = idx.pos_lists[a]
                sel = rng.choice(pos, min(P, len(pos)), replace=False)
                for m in [a, *sel.tolist()]:
                    rows.append(int(m)); groups.append(gi)
            if len(rows) < 2:
                continue
            gids = torch.tensor(groups, device=device)
            pos_mask = gids[:, None] == gids[None, :]
            opt.zero_grad()
            z = enc(Xt[rows])
            loss = criterion(z, pos_mask)
            loss.backward(); opt.step()
            running += loss.item(); nb += 1
        history.append(running / max(nb, 1))
    return enc, history


@torch.no_grad()
def encode(enc, X, device, batch_size=1024):
    enc.eval()
    out = []
    for s in range(0, len(X), batch_size):
        xb = torch.from_numpy(X[s:s + batch_size]).to(device)
        out.append(enc(xb).cpu().numpy())
    return np.concatenate(out, axis=0)


# --------------------------------------------------------------------------- #
# Stage B / baseline: supervised head on fixed features
# --------------------------------------------------------------------------- #
def train_head(kind, Xtr, ytr, Xva, yva, dataset, cfg, device=None):
    """Train an MLP/TabResNet head on given feature matrices. Early-stops on val."""
    device = device or resolve_device(cfg.get("device", "auto"))
    torch.manual_seed(cfg.get("seed", 0))
    out_dim = head_out_dim(dataset)
    model = build_head(kind, Xtr.shape[1], out_dim, cfg).to(device)
    loss_fn = _task_loss(dataset.task_type)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.get("head_lr", 1e-3),
                            weight_decay=cfg.get("weight_decay", 1e-4))

    ytr_t = torch.from_numpy(ytr).to(device)
    Xtr_t = torch.from_numpy(Xtr).to(device)
    ds = TensorDataset(Xtr_t, ytr_t)
    loader = DataLoader(ds, batch_size=cfg.get("batch_size", 256), shuffle=True)

    best_val, best_state, patience, bad = np.inf, None, cfg.get("patience", 16), 0
    for _ in range(cfg.get("head_epochs", 100)):
        model.train()
        for xb, yb in loader:
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward(); opt.step()
        val = _eval_head_loss(model, Xva, yva, dataset, loss_fn, device)
        if val < best_val - 1e-5:
            best_val, best_state, bad = val, \
                {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model


@torch.no_grad()
def _eval_head_loss(model, X, y, dataset, loss_fn, device):
    model.eval()
    xb = torch.from_numpy(X).to(device)
    yb = torch.from_numpy(y).to(device)
    return float(loss_fn(model(xb), yb).item())


@torch.no_grad()
def predict_head(model, X, dataset, device):
    model.eval()
    xb = torch.from_numpy(X).to(device)
    out = model(xb)
    if dataset.task_type == "classification":
        return F.softmax(out, dim=1).cpu().numpy()
    return out.squeeze(-1).cpu().numpy()


# --------------------------------------------------------------------------- #
# Joint end-to-end regime
# --------------------------------------------------------------------------- #
def train_joint(head_kind, Xtr, ytr, Xva, yva, kernel, dataset, cfg, device=None):
    """Encoder+head trained together: task_loss + lambda*contrastive_loss."""
    device = device or resolve_device(cfg.get("device", "auto"))
    torch.manual_seed(cfg.get("seed", 0))
    enc = SiameseEncoder(
        Xtr.shape[1], hidden_dims=tuple(cfg.get("enc_hidden", (256, 128))),
        embedding_dim=cfg.get("embedding_dim", 64),
        dropout=cfg.get("enc_dropout", 0.1)).to(device)
    head = build_head(head_kind, cfg.get("embedding_dim", 64),
                      head_out_dim(dataset), cfg).to(device)
    model = JointModel(enc, head).to(device)
    task_loss = _task_loss(dataset.task_type)
    loss_name = cfg.get("contrastive_loss", "infonce")
    contrast, regime = build_contrastive(
        loss_name, cfg.get("temperature", 0.1),
        dim=cfg.get("embedding_dim", 64), margin=cfg.get("margin", 1.0))
    if regime not in ("anchorpos",):
        raise ValueError(
            f"joint regime supports the anchor/positive losses "
            f"(infonce/aninfonce/clip_infonce/contrastive/triplet); "
            f"'{loss_name}' is only available in two-stage.")
    contrast = contrast.to(device)
    lam = cfg.get("lambda_contrast", 0.5)
    opt = torch.optim.AdamW(
        list(model.parameters()) + list(contrast.parameters()),
        lr=cfg.get("head_lr", 1e-3), weight_decay=cfg.get("weight_decay", 1e-4))

    # Contrastive pairs from the kernel (anchor/positive indices on train rows).
    leaves = kernel.leaves(Xtr)
    idx = SampledPositiveIndex(kernel, leaves, pos_threshold=cfg.get("pos_threshold", 0.6),
                               max_pos=cfg.get("max_pos", 50), seed=cfg.get("seed", 0))
    ap = AnchorPositiveDataset(Xtr, idx, seed=cfg.get("seed", 0))
    # drop_last=False + an infinite cycle so a small/sparse positive set (e.g.
    # under many noise features) still contributes without emptying the loader.
    ap_gen = None
    if len(ap) > 0 and lam > 0:
        ap_loader = DataLoader(ap, batch_size=cfg.get("batch_size", 256),
                               shuffle=True, drop_last=False)
        ap_gen = _cycle(ap_loader)

    Xtr_t, ytr_t = torch.from_numpy(Xtr).to(device), torch.from_numpy(ytr).to(device)
    sup_loader = DataLoader(TensorDataset(Xtr_t, ytr_t),
                            batch_size=cfg.get("batch_size", 256), shuffle=True)

    best_val, best_state, patience, bad = np.inf, None, cfg.get("patience", 16), 0
    for _ in range(cfg.get("head_epochs", 100)):
        model.train()
        for xb, yb in sup_loader:
            opt.zero_grad()
            out, _ = model(xb)
            loss = task_loss(out, yb)
            if ap_gen is not None:
                xi, xj = next(ap_gen)
                xi, xj = xi.to(device), xj.to(device)
                loss = loss + lam * apply_pair_loss(loss_name, contrast,
                                                    enc(xi), enc(xj))
            loss.backward(); opt.step()
        val = _eval_joint_loss(model, Xva, yva, dataset, task_loss, device)
        if val < best_val - 1e-5:
            best_val, best_state, bad = val, \
                {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model


@torch.no_grad()
def _eval_joint_loss(model, X, y, dataset, loss_fn, device):
    model.eval()
    out, _ = model(torch.from_numpy(X).to(device))
    return float(loss_fn(out, torch.from_numpy(y).to(device)).item())


@torch.no_grad()
def predict_joint(model, X, dataset, device):
    model.eval()
    out, _ = model(torch.from_numpy(X).to(device))
    if dataset.task_type == "classification":
        return F.softmax(out, dim=1).cpu().numpy()
    return out.squeeze(-1).cpu().numpy()
