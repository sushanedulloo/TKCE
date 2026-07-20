"""Deep TKCE-joint training-dynamics study on ONE balanced classification dataset.

Purpose (per-epoch trends, not benchmarking): take a single balanced binary
dataset, make the Siamese encoder and the TabResNet head much DEEPER, train for
many epochs with a SLOW learning rate, and record/plot how the loss and the
AUC/accuracy evolve on BOTH the training and validation sets.

Unlike the main suite, there is NO early stopping here: we deliberately train the
full number of epochs so the whole trend (including any overfitting) is visible.

Outputs:
  <csv>/deep_joint_<dataset>.csv   per-epoch metrics
  <out>/deep_joint_<dataset>.png   4-panel figure (loss, AUC, accuracy, contrastive)

Usage (defaults follow the "deep + long + slow" recipe):
  python run_deep_joint.py --task 361070 --epochs 600 --lr 1e-4 --device auto
"""

from __future__ import annotations

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from tkce.data import load_task
from tkce.kernels import build_kernel
from tkce.losses import apply_pair_loss, build_contrastive
from tkce.metrics import clf_metrics
from tkce.models import JointModel, SiameseEncoder, build_head, head_out_dim
from tkce.pairs import AnchorPositiveDataset, SampledPositiveIndex
from tkce.train import resolve_device


def _cycle(loader):
    while True:
        for b in loader:
            yield b


@torch.no_grad()
def _eval_split(model, X, y, device, batch_size=4096):
    """Return (cross-entropy loss, AUC, accuracy) on a split."""
    model.eval()
    losses, probas = [], []
    for s in range(0, len(X), batch_size):
        xb = torch.from_numpy(X[s:s + batch_size]).to(device)
        yb = torch.from_numpy(y[s:s + batch_size]).to(device)
        out, _ = model(xb)
        losses.append(F.cross_entropy(out, yb, reduction="sum").item())
        probas.append(F.softmax(out, dim=1).cpu().numpy())
    proba = np.concatenate(probas, axis=0)
    m = clf_metrics(y, proba)
    return sum(losses) / len(X), m["auc"], m["accuracy"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", type=int, default=361070, help="balanced binary clf (eye_movements)")
    ap.add_argument("--epochs", type=int, default=600)
    ap.add_argument("--lr", type=float, default=1e-4, help="slow learning rate")
    ap.add_argument("--head", default="tabresnet", choices=["tabresnet", "mlp"])
    # --- deep encoder ---
    ap.add_argument("--enc-width", type=int, default=512)
    ap.add_argument("--enc-depth", type=int, default=6)
    ap.add_argument("--embedding-dim", type=int, default=128)
    ap.add_argument("--enc-dropout", type=float, default=0.1)
    # --- deep TabResNet head ---
    ap.add_argument("--d", type=int, default=256)
    ap.add_argument("--d-hidden", type=int, default=512)
    ap.add_argument("--n-blocks", type=int, default=8)
    ap.add_argument("--dropout", type=float, default=0.1)
    # --- contrastive / kernel ---
    ap.add_argument("--lambda-contrast", type=float, default=0.5)
    ap.add_argument("--contrastive-loss", default="infonce")
    ap.add_argument("--temperature", type=float, default=0.1)
    ap.add_argument("--pos-threshold", type=float, default=0.6)
    ap.add_argument("--k-n-estimators", type=int, default=200)
    ap.add_argument("--k-max-depth", type=int, default=4)
    # --- misc ---
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-rows", type=int, default=16000)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", default="paper/figures")
    ap.add_argument("--csv", default="results/analysis")
    args = ap.parse_args()

    device = resolve_device(args.device)
    os.makedirs(args.out, exist_ok=True); os.makedirs(args.csv, exist_ok=True)
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    ds = load_task(args.task, seed=args.seed, max_rows=args.max_rows)
    if ds.task_type != "classification":
        raise SystemExit(f"{ds.name} is not a classification dataset.")
    counts = np.bincount(ds.y_train)
    print(f"[deep] {ds.name} | classes={ds.n_classes} | train class balance="
          f"{np.round(counts / counts.sum(), 3).tolist()} | features={ds.n_features}")
    print(f"[deep] device={device} epochs={args.epochs} lr={args.lr} "
          f"encoder={args.enc_depth}x{args.enc_width} head={args.head}"
          f"(n_blocks={args.n_blocks}) lambda={args.lambda_contrast}")

    # ---- deep model: Siamese encoder + deep head, trained JOINTLY ----
    enc = SiameseEncoder(ds.n_features,
                         hidden_dims=tuple([args.enc_width] * args.enc_depth),
                         embedding_dim=args.embedding_dim,
                         dropout=args.enc_dropout).to(device)
    head_cfg = dict(hidden_dims=(args.d, args.d), dropout=args.dropout,
                    d=args.d, d_hidden=args.d_hidden, n_blocks=args.n_blocks)
    head = build_head(args.head, args.embedding_dim, head_out_dim(ds), head_cfg).to(device)
    model = JointModel(enc, head).to(device)
    n_par = sum(p.numel() for p in model.parameters())
    print(f"[deep] trainable parameters: {n_par:,}")

    # ---- contrastive signal from the tree kernel ----
    contrast, regime = build_contrastive(args.contrastive_loss, args.temperature,
                                         dim=args.embedding_dim)
    contrast = contrast.to(device)
    kern = build_kernel("gbt", ds.X_train, ds.y_train, ds.task_type,
                        {"n_estimators": args.k_n_estimators,
                         "max_depth": args.k_max_depth, "random_state": args.seed})
    leaves = kern.leaves(ds.X_train)
    idx = SampledPositiveIndex(kern, leaves, pos_threshold=args.pos_threshold,
                               max_pos=50, seed=args.seed)
    ap_ds = AnchorPositiveDataset(ds.X_train, idx, seed=args.seed)
    print(f"[deep] anchors with >=1 positive: {len(ap_ds)}/{idx.n}")
    ap_gen = None
    if len(ap_ds) > 0 and args.lambda_contrast > 0:
        ap_gen = _cycle(DataLoader(ap_ds, batch_size=args.batch_size,
                                   shuffle=True, drop_last=False))

    opt = torch.optim.AdamW(list(model.parameters()) + list(contrast.parameters()),
                            lr=args.lr, weight_decay=args.weight_decay)

    Xtr_t = torch.from_numpy(ds.X_train).to(device)
    ytr_t = torch.from_numpy(ds.y_train).to(device)
    sup_loader = DataLoader(TensorDataset(Xtr_t, ytr_t),
                            batch_size=args.batch_size, shuffle=True)

    # ---- long, slow training with NO early stopping ----
    hist = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        tot = task_sum = con_sum = 0.0
        nb = 0
        for xb, yb in sup_loader:
            opt.zero_grad()
            out, _ = model(xb)
            t_loss = F.cross_entropy(out, yb)
            loss = t_loss
            c_val = 0.0
            if ap_gen is not None:
                xi, xj = next(ap_gen)
                c_loss = apply_pair_loss(args.contrastive_loss, contrast,
                                         enc(xi.to(device)), enc(xj.to(device)))
                loss = loss + args.lambda_contrast * c_loss
                c_val = c_loss.item()
            loss.backward(); opt.step()
            tot += loss.item(); task_sum += t_loss.item(); con_sum += c_val; nb += 1

        tr_loss, tr_auc, tr_acc = _eval_split(model, ds.X_train, ds.y_train, device)
        va_loss, va_auc, va_acc = _eval_split(model, ds.X_val, ds.y_val, device)
        hist.append(dict(epoch=epoch,
                         batch_total_loss=tot / nb, batch_task_loss=task_sum / nb,
                         batch_contrastive_loss=con_sum / nb,
                         train_loss=tr_loss, val_loss=va_loss,
                         train_auc=tr_auc, val_auc=va_auc,
                         train_acc=tr_acc, val_acc=va_acc))
        if epoch % max(1, args.epochs // 20) == 0 or epoch == 1:
            print(f"[deep] epoch {epoch:4d}/{args.epochs}  "
                  f"train_loss={tr_loss:.4f} val_loss={va_loss:.4f}  "
                  f"train_auc={tr_auc:.4f} val_auc={va_auc:.4f}", flush=True)

    df = pd.DataFrame(hist)
    csv_path = os.path.join(args.csv, f"deep_joint_{ds.name}.csv")
    df.to_csv(csv_path, index=False)

    te_loss, te_auc, te_acc = _eval_split(model, ds.X_test, ds.y_test, device)
    best = df.loc[df.val_auc.idxmax()]
    print(f"\n[deep] best val AUC {best.val_auc:.4f} at epoch {int(best.epoch)} "
          f"(val acc {best.val_acc:.4f})")
    print(f"[deep] final test AUC {te_auc:.4f} | acc {te_acc:.4f} | loss {te_loss:.4f}")

    # ---- figure: loss / AUC / accuracy / contrastive, train vs val ----
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    e = df.epoch

    axes[0, 0].plot(e, df.train_loss, label="train")
    axes[0, 0].plot(e, df.val_loss, label="validation")
    axes[0, 0].set_title("Task loss (cross-entropy)"); axes[0, 0].set_xlabel("epoch")
    axes[0, 0].set_ylabel("loss")

    axes[0, 1].plot(e, df.train_auc, label="train")
    axes[0, 1].plot(e, df.val_auc, label="validation")
    axes[0, 1].axvline(int(best.epoch), ls="--", c="grey", lw=1,
                       label=f"best val epoch {int(best.epoch)}")
    axes[0, 1].set_title("AUC"); axes[0, 1].set_xlabel("epoch"); axes[0, 1].set_ylabel("AUC")

    axes[1, 0].plot(e, df.train_acc, label="train")
    axes[1, 0].plot(e, df.val_acc, label="validation")
    axes[1, 0].set_title("Accuracy"); axes[1, 0].set_xlabel("epoch")
    axes[1, 0].set_ylabel("accuracy")

    axes[1, 1].plot(e, df.batch_contrastive_loss, c="tab:green", label="contrastive")
    axes[1, 1].plot(e, df.batch_task_loss, c="tab:orange", label="task (batch avg)")
    axes[1, 1].set_title("Training loss components"); axes[1, 1].set_xlabel("epoch")
    axes[1, 1].set_ylabel("loss")

    for a in axes.ravel():
        a.grid(alpha=0.3); a.legend(fontsize=8)
    fig.suptitle(f"Deep TKCE-joint on {ds.name} (balanced binary) — "
                 f"encoder {args.enc_depth}x{args.enc_width}, {args.head} "
                 f"n_blocks={args.n_blocks}, lr={args.lr}, {args.epochs} epochs",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    png = os.path.join(args.out, f"deep_joint_{ds.name}.png")
    fig.savefig(png, dpi=150, bbox_inches="tight")
    print(f"[deep] figure -> {png}\n[deep] per-epoch metrics -> {csv_path}")


if __name__ == "__main__":
    main()
