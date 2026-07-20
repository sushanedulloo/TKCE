"""Deep TKCE-joint training-dynamics study over MULTIPLE SHUFFLES of one dataset.

Purpose: take a single balanced binary dataset, make the Siamese encoder and the
TabResNet head much DEEPER, train for many epochs with a SLOW learning rate and
NO early stopping, and do this over several random train/val/test SHUFFLES.

This answers two questions at once:
  1. How do loss / AUC / accuracy evolve per epoch on train vs validation?
  2. Is the test score a genuine ceiling for this dataset, or an artifact of one
     lucky/unlucky split? (Each shuffle re-splits the data with a new seed.)

Outputs:
  <csv>/deep_joint_<data>_epochs.csv  per-epoch metrics for every shuffle
  <csv>/deep_joint_<data>_test.csv    final test metrics per shuffle
  <out>/deep_joint_<data>.png         4 panels: loss, AUC, accuracy (mean +/- band
                                      across shuffles) and per-shuffle test AUC

Usage:
  python run_deep_joint.py --task 361070 --seeds 0,1,2,3,4 --epochs 600 --lr 1e-4
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
    tot, probas = 0.0, []
    for s in range(0, len(X), batch_size):
        xb = torch.from_numpy(X[s:s + batch_size]).to(device)
        yb = torch.from_numpy(y[s:s + batch_size]).to(device)
        out, _ = model(xb)
        tot += F.cross_entropy(out, yb, reduction="sum").item()
        probas.append(F.softmax(out, dim=1).cpu().numpy())
    m = clf_metrics(y, np.concatenate(probas, axis=0))
    return tot / len(X), m["auc"], m["accuracy"]


def run_one_shuffle(seed, args, device):
    """Train the deep joint model on ONE random split. Returns (history, test)."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    ds = load_task(args.task, seed=seed, max_rows=args.max_rows)   # new shuffle
    if ds.task_type != "classification":
        raise SystemExit(f"{ds.name} is not a classification dataset.")

    enc = SiameseEncoder(ds.n_features,
                         hidden_dims=tuple([args.enc_width] * args.enc_depth),
                         embedding_dim=args.embedding_dim,
                         dropout=args.enc_dropout).to(device)
    head_cfg = dict(hidden_dims=(args.d, args.d), dropout=args.dropout,
                    d=args.d, d_hidden=args.d_hidden, n_blocks=args.n_blocks)
    head = build_head(args.head, args.embedding_dim, head_out_dim(ds), head_cfg).to(device)
    model = JointModel(enc, head).to(device)

    contrast, _ = build_contrastive(args.contrastive_loss, args.temperature,
                                    dim=args.embedding_dim)
    contrast = contrast.to(device)
    kern = build_kernel("gbt", ds.X_train, ds.y_train, ds.task_type,
                        {"n_estimators": args.k_n_estimators,
                         "max_depth": args.k_max_depth, "random_state": seed})
    idx = SampledPositiveIndex(kern, kern.leaves(ds.X_train),
                               pos_threshold=args.pos_threshold, max_pos=50, seed=seed)
    ap_ds = AnchorPositiveDataset(ds.X_train, idx, seed=seed)
    ap_gen = None
    if len(ap_ds) > 0 and args.lambda_contrast > 0:
        ap_gen = _cycle(DataLoader(ap_ds, batch_size=args.batch_size,
                                   shuffle=True, drop_last=False))

    opt = torch.optim.AdamW(list(model.parameters()) + list(contrast.parameters()),
                            lr=args.lr, weight_decay=args.weight_decay)
    sup_loader = DataLoader(
        TensorDataset(torch.from_numpy(ds.X_train).to(device),
                      torch.from_numpy(ds.y_train).to(device)),
        batch_size=args.batch_size, shuffle=True)

    # ---- loss balancing: measure the two losses' starting magnitudes ----
    lam_eff = args.lambda_contrast
    t0 = c0 = float("nan")
    with torch.no_grad():
        xb0, yb0 = next(iter(sup_loader))
        t0 = F.cross_entropy(model(xb0)[0], yb0).item()
        if ap_gen is not None:
            xi0, xj0 = next(ap_gen)
            c0 = apply_pair_loss(args.contrastive_loss, contrast,
                                 enc(xi0.to(device)), enc(xj0.to(device))).item()
    if args.balance_losses and np.isfinite(c0) and c0 > 1e-8:
        lam_eff = args.lambda_contrast * (t0 / c0)

    print(f"[shuffle seed={seed}] {ds.name}: train={len(ds.X_train)} "
          f"val={len(ds.X_val)} test={len(ds.X_test)} | anchors with positives="
          f"{len(ap_ds)}/{idx.n}", flush=True)
    print(f"  initial magnitudes: task={t0:.3f}  contrastive={c0:.3f}  "
          f"(ratio {c0 / max(t0, 1e-8):.1f}x) -> lambda_effective={lam_eff:.4f}"
          f"{'  [auto-balanced]' if args.balance_losses else ''}", flush=True)

    hist = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        task_sum = con_sum = 0.0
        nb = 0
        for xb, yb in sup_loader:
            opt.zero_grad()
            out, _ = model(xb)
            t_loss = F.cross_entropy(out, yb)
            loss, c_val = t_loss, 0.0
            if ap_gen is not None:
                xi, xj = next(ap_gen)
                c_loss = apply_pair_loss(args.contrastive_loss, contrast,
                                         enc(xi.to(device)), enc(xj.to(device)))
                loss = loss + lam_eff * c_loss
                c_val = c_loss.item()
            loss.backward(); opt.step()
            task_sum += t_loss.item(); con_sum += c_val; nb += 1

        tr_loss, tr_auc, tr_acc = _eval_split(model, ds.X_train, ds.y_train, device)
        va_loss, va_auc, va_acc = _eval_split(model, ds.X_val, ds.y_val, device)
        hist.append(dict(seed=seed, epoch=epoch, lam_eff=lam_eff,
                         batch_task_loss=task_sum / nb,
                         batch_contrastive_loss=con_sum / nb,
                         weighted_contrastive=lam_eff * con_sum / nb,
                         train_loss=tr_loss, val_loss=va_loss,
                         train_auc=tr_auc, val_auc=va_auc,
                         train_acc=tr_acc, val_acc=va_acc))
        if epoch % max(1, args.epochs // 10) == 0 or epoch == 1:
            print(f"  [seed {seed}] epoch {epoch:4d}/{args.epochs} "
                  f"train_auc={tr_auc:.4f} val_auc={va_auc:.4f} "
                  f"val_loss={va_loss:.4f}", flush=True)

    te_loss, te_auc, te_acc = _eval_split(model, ds.X_test, ds.y_test, device)
    h = pd.DataFrame(hist)
    best = h.loc[h.val_auc.idxmax()]
    test = dict(seed=seed, dataset=ds.name, test_auc=te_auc, test_acc=te_acc,
                test_loss=te_loss, best_val_auc=best.val_auc,
                best_val_epoch=int(best.epoch))
    print(f"  [seed {seed}] TEST auc={te_auc:.4f} acc={te_acc:.4f} "
          f"(best val auc={best.val_auc:.4f} @ epoch {int(best.epoch)})\n", flush=True)
    return h, test, ds.name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", type=int, default=361070, help="balanced binary clf")
    ap.add_argument("--seeds", default="0,1,2,3,4",
                    help="comma-separated shuffles (each = a new random split)")
    ap.add_argument("--epochs", type=int, default=600)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--head", default="tabresnet", choices=["tabresnet", "mlp"])
    ap.add_argument("--enc-width", type=int, default=512)
    ap.add_argument("--enc-depth", type=int, default=6)
    ap.add_argument("--embedding-dim", type=int, default=128)
    ap.add_argument("--enc-dropout", type=float, default=0.1)
    ap.add_argument("--d", type=int, default=256)
    ap.add_argument("--d-hidden", type=int, default=512)
    ap.add_argument("--n-blocks", type=int, default=8)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--lambda-contrast", type=float, default=0.5)
    ap.add_argument("--balance-losses", action="store_true",
                    help="rescale lambda so the task and contrastive terms START "
                         "with equal magnitude (fixes the 'task loss is subdued' "
                         "problem when InfoNCE is ~8x larger than cross-entropy)")
    ap.add_argument("--contrastive-loss", default="infonce")
    ap.add_argument("--temperature", type=float, default=0.1)
    ap.add_argument("--pos-threshold", type=float, default=0.6)
    ap.add_argument("--k-n-estimators", type=int, default=200)
    ap.add_argument("--k-max-depth", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--max-rows", type=int, default=16000)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", default="paper/figures")
    ap.add_argument("--csv", default="results/analysis")
    args = ap.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    device = resolve_device(args.device)
    os.makedirs(args.out, exist_ok=True); os.makedirs(args.csv, exist_ok=True)
    print(f"[deep] {len(seeds)} shuffles {seeds} | epochs={args.epochs} lr={args.lr} "
          f"| encoder {args.enc_depth}x{args.enc_width} | {args.head} "
          f"n_blocks={args.n_blocks} | device={device}\n", flush=True)

    hists, tests, name = [], [], None
    for s in seeds:
        h, t, name = run_one_shuffle(s, args, device)
        hists.append(h); tests.append(t)

    df = pd.concat(hists, ignore_index=True)
    tdf = pd.DataFrame(tests)
    df.to_csv(os.path.join(args.csv, f"deep_joint_{name}_epochs.csv"), index=False)
    tdf.to_csv(os.path.join(args.csv, f"deep_joint_{name}_test.csv"), index=False)

    # ---------------- summary ----------------
    print("=" * 62)
    print(f"TEST RESULTS ACROSS {len(seeds)} SHUFFLES — {name}")
    print("=" * 62)
    print(tdf[["seed", "test_auc", "test_acc", "best_val_auc", "best_val_epoch"]]
          .to_string(index=False))
    m, sd = tdf.test_auc.mean(), tdf.test_auc.std(ddof=0)
    ma, sda = tdf.test_acc.mean(), tdf.test_acc.std(ddof=0)
    print(f"\ntest AUC      : {m:.4f} +/- {sd:.4f}   "
          f"(min {tdf.test_auc.min():.4f}, max {tdf.test_auc.max():.4f})")
    print(f"test accuracy : {ma:.4f} +/- {sda:.4f}")
    print("=> the spread across shuffles shows whether this is a real ceiling.")

    # ---------------- figure ----------------
    g = df.groupby("epoch")
    mean, std = g.mean(numeric_only=True), g.std(numeric_only=True).fillna(0.0)
    e = mean.index

    def band(ax, col, label, color):
        ax.plot(e, mean[col], label=label, color=color)
        ax.fill_between(e, mean[col] - std[col], mean[col] + std[col],
                        alpha=0.15, color=color)

    fig, axes = plt.subplots(2, 3, figsize=(18, 8.5))

    # --- (a) TASK loss, own scale ---
    band(axes[0, 0], "train_loss", "train", "tab:blue")
    band(axes[0, 0], "val_loss", "validation", "tab:orange")
    axes[0, 0].set_title("TASK loss (cross-entropy)")
    axes[0, 0].set_xlabel("epoch"); axes[0, 0].set_ylabel("loss")

    # --- (b) CONTRASTIVE loss, own scale ---
    band(axes[0, 1], "batch_contrastive_loss", "contrastive (raw)", "tab:green")
    axes[0, 1].set_title("CONTRASTIVE loss (own scale)")
    axes[0, 1].set_xlabel("epoch"); axes[0, 1].set_ylabel("loss")

    # --- (c) the imbalance: what each term CONTRIBUTES to the total ---
    band(axes[0, 2], "batch_task_loss", "task", "tab:orange")
    band(axes[0, 2], "weighted_contrastive",
         f"$\\lambda\\cdot$contrastive ($\\lambda$={df.lam_eff.iloc[0]:.3g})", "tab:green")
    share = df.weighted_contrastive.mean() / max(
        df.weighted_contrastive.mean() + df.batch_task_loss.mean(), 1e-9)
    axes[0, 2].set_title(f"Contribution to the fused loss\n"
                         f"(contrastive = {share:.0%} of total)")
    axes[0, 2].set_xlabel("epoch"); axes[0, 2].set_ylabel("loss contribution")

    band(axes[1, 0], "train_auc", "train", "tab:blue")
    band(axes[1, 0], "val_auc", "validation", "tab:orange")
    axes[1, 0].axhline(m, ls="--", c="tab:red", lw=1.4,
                       label=f"mean TEST AUC = {m:.3f}")
    axes[1, 0].set_title("AUC (mean $\\pm$ std over shuffles)")
    axes[1, 0].set_xlabel("epoch"); axes[1, 0].set_ylabel("AUC")

    band(axes[1, 1], "train_acc", "train", "tab:blue")
    band(axes[1, 1], "val_acc", "validation", "tab:orange")
    axes[1, 1].axhline(ma, ls="--", c="tab:red", lw=1.4,
                       label=f"mean TEST acc = {ma:.3f}")
    axes[1, 1].set_title("Accuracy (mean $\\pm$ std over shuffles)")
    axes[1, 1].set_xlabel("epoch"); axes[1, 1].set_ylabel("accuracy")

    axes[1, 2].bar(range(len(tdf)), tdf.test_auc, color="steelblue",
                   edgecolor="white")
    axes[1, 2].axhline(m, ls="--", c="tab:red", lw=1.4, label=f"mean {m:.3f}")
    axes[1, 2].fill_between([-0.5, len(tdf) - 0.5], m - sd, m + sd,
                            color="tab:red", alpha=0.12, label=f"$\\pm$ std {sd:.3f}")
    axes[1, 2].axhline(0.5, ls=":", c="grey", lw=1.2, label="random (0.5)")
    axes[1, 2].set_xticks(range(len(tdf)))
    axes[1, 2].set_xticklabels([f"seed {s}" for s in tdf.seed], fontsize=8)
    axes[1, 2].set_xlim(-0.5, len(tdf) - 0.5)
    # adaptive limits so no bar is ever clipped
    lo = max(0.0, min(0.48, tdf.test_auc.min() - 0.05))
    hi = min(1.0, max(0.75, tdf.test_auc.max() + 0.05))
    axes[1, 2].set_ylim(lo, hi)
    axes[1, 2].set_title("Final TEST AUC per shuffle (is it a real ceiling?)")
    axes[1, 2].set_ylabel("test AUC")

    for a in axes.ravel():
        a.grid(alpha=0.3); a.legend(fontsize=8)
    fig.suptitle(f"Deep TKCE-joint on {name} — {len(seeds)} shuffles, "
                 f"encoder {args.enc_depth}x{args.enc_width}, {args.head} "
                 f"n_blocks={args.n_blocks}, lr={args.lr}, {args.epochs} epochs",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    png = os.path.join(args.out, f"deep_joint_{name}.png")
    fig.savefig(png, dpi=150, bbox_inches="tight")
    print(f"\n[deep] figure -> {png}")
    print(f"[deep] per-epoch data -> {args.csv}/deep_joint_{name}_epochs.csv")
    print(f"[deep] test summary  -> {args.csv}/deep_joint_{name}_test.csv")


if __name__ == "__main__":
    main()
