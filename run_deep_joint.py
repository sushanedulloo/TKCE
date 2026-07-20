"""Deep TKCE-joint study: LOSSES x SHUFFLES on one balanced classification dataset.

Deep Siamese encoder + deep TabResNet, many epochs, slow learning rate, NO early
stopping. Sweeps several contrastive losses, and repeats each over several random
train/val/test shuffles, so you can see:

  * which contrastive loss gives the best downstream score,
  * whether the test score is a genuine ceiling or a lucky/unlucky split,
  * how loss / AUC / accuracy evolve per epoch on train vs validation,
  * how badly the contrastive term dominates the fused loss (per loss, since the
    losses have very different natural magnitudes).

NOTE: the joint regime supports the in-batch anchor/positive family only
(infonce, aninfonce, clip_infonce, contrastive, triplet). `supcon` and
`kernel_regression` need different batch structures and are two-stage only; if
passed they are skipped with a message.

Outputs (in <csv> / <out>):
  deep_joint_<data>_epochs.csv  per-epoch metrics for every (loss, shuffle)
  deep_joint_<data>_test.csv    final test metrics for every (loss, shuffle)
  deep_joint_<data>.png         6-panel comparison figure

Usage:
  python run_deep_joint.py --task 361070 --seeds 0,1,2 \
      --losses infonce,aninfonce,clip_infonce,contrastive,triplet \
      --epochs 600 --lr 1e-4
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

JOINT_LOSSES = ["infonce", "aninfonce", "clip_infonce", "contrastive", "triplet"]


def _cycle(loader):
    while True:
        for b in loader:
            yield b


@torch.no_grad()
def _eval_split(model, X, y, device, batch_size=4096):
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


def run_one(loss_name, seed, args, device):
    """Train the deep joint model with one loss on one shuffle."""
    torch.manual_seed(seed); np.random.seed(seed)
    ds = load_task(args.task, seed=seed, max_rows=args.max_rows)
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

    contrast, regime = build_contrastive(loss_name, args.temperature,
                                         dim=args.embedding_dim, margin=args.margin)
    if regime != "anchorpos":
        raise ValueError(f"'{loss_name}' is not supported in the joint regime "
                         f"(two-stage only)")
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

    # measure starting magnitudes -> optional auto-balance of the fused loss
    lam_eff, t0, c0 = args.lambda_contrast, float("nan"), float("nan")
    with torch.no_grad():
        xb0, yb0 = next(iter(sup_loader))
        t0 = F.cross_entropy(model(xb0)[0], yb0).item()
        if ap_gen is not None:
            xi0, xj0 = next(ap_gen)
            c0 = apply_pair_loss(loss_name, contrast,
                                 enc(xi0.to(device)), enc(xj0.to(device))).item()
    if args.balance_losses and np.isfinite(c0) and c0 > 1e-8:
        lam_eff = args.lambda_contrast * (t0 / c0)
    print(f"  [{loss_name} | seed {seed}] task0={t0:.3f} contrastive0={c0:.3f} "
          f"(ratio {c0/max(t0,1e-8):.1f}x) lambda_eff={lam_eff:.4f}", flush=True)

    hist = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        task_sum = con_sum = 0.0; nb = 0
        for xb, yb in sup_loader:
            opt.zero_grad()
            out, _ = model(xb)
            t_loss = F.cross_entropy(out, yb)
            loss, c_val = t_loss, 0.0
            if ap_gen is not None:
                xi, xj = next(ap_gen)
                c_loss = apply_pair_loss(loss_name, contrast,
                                         enc(xi.to(device)), enc(xj.to(device)))
                loss = loss + lam_eff * c_loss
                c_val = c_loss.item()
            loss.backward(); opt.step()
            task_sum += t_loss.item(); con_sum += c_val; nb += 1

        tr_loss, tr_auc, tr_acc = _eval_split(model, ds.X_train, ds.y_train, device)
        va_loss, va_auc, va_acc = _eval_split(model, ds.X_val, ds.y_val, device)
        hist.append(dict(loss=loss_name, seed=seed, epoch=epoch, lam_eff=lam_eff,
                         batch_task_loss=task_sum / nb,
                         batch_contrastive_loss=con_sum / nb,
                         weighted_contrastive=lam_eff * con_sum / nb,
                         train_loss=tr_loss, val_loss=va_loss,
                         train_auc=tr_auc, val_auc=va_auc,
                         train_acc=tr_acc, val_acc=va_acc))
        if epoch % max(1, args.epochs // 6) == 0:
            print(f"    epoch {epoch:4d}/{args.epochs} train_auc={tr_auc:.4f} "
                  f"val_auc={va_auc:.4f}", flush=True)

    te_loss, te_auc, te_acc = _eval_split(model, ds.X_test, ds.y_test, device)
    h = pd.DataFrame(hist)
    best = h.loc[h.val_auc.idxmax()]
    test = dict(loss=loss_name, seed=seed, dataset=ds.name, test_auc=te_auc,
                test_acc=te_acc, test_loss=te_loss, train_auc_final=h.train_auc.iloc[-1],
                best_val_auc=best.val_auc, best_val_epoch=int(best.epoch),
                lam_eff=lam_eff, contrastive0=c0, task0=t0)
    print(f"  [{loss_name} | seed {seed}] TEST auc={te_auc:.4f} acc={te_acc:.4f}\n",
          flush=True)
    return h, test, ds.name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", type=int, default=361070)
    ap.add_argument("--seeds", default="0,1,2", help="shuffles (new split each)")
    ap.add_argument("--losses", default=",".join(JOINT_LOSSES),
                    help="contrastive losses to compare (joint-compatible only)")
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
                    help="rescale lambda per loss so task and contrastive start equal")
    ap.add_argument("--temperature", type=float, default=0.1)
    ap.add_argument("--margin", type=float, default=1.0)
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
    losses = [l.strip() for l in args.losses.split(",")]
    device = resolve_device(args.device)
    os.makedirs(args.out, exist_ok=True); os.makedirs(args.csv, exist_ok=True)
    print(f"[deep] {len(losses)} losses x {len(seeds)} shuffles = "
          f"{len(losses)*len(seeds)} runs | epochs={args.epochs} lr={args.lr} | "
          f"encoder {args.enc_depth}x{args.enc_width} | {args.head} "
          f"n_blocks={args.n_blocks} | device={device}\n", flush=True)

    hists, tests, name = [], [], None
    for L in losses:
        print(f"===== loss: {L} =====", flush=True)
        for s in seeds:
            try:
                h, t, name = run_one(L, s, args, device)
                hists.append(h); tests.append(t)
            except ValueError as e:
                print(f"  SKIP {L}: {e}\n", flush=True)
                break

    if not tests:
        raise SystemExit("no runs completed")
    df = pd.concat(hists, ignore_index=True)
    tdf = pd.DataFrame(tests)
    df.to_csv(os.path.join(args.csv, f"deep_joint_{name}_epochs.csv"), index=False)
    tdf.to_csv(os.path.join(args.csv, f"deep_joint_{name}_test.csv"), index=False)

    # ---------------- summary ----------------
    summ = (tdf.groupby("loss")
              .agg(test_auc_mean=("test_auc", "mean"), test_auc_std=("test_auc", "std"),
                   test_acc_mean=("test_acc", "mean"), n=("test_auc", "size"),
                   lam_eff=("lam_eff", "mean"), contrastive0=("contrastive0", "mean"))
              .sort_values("test_auc_mean", ascending=False))
    summ["test_auc_std"] = summ["test_auc_std"].fillna(0.0)
    print("=" * 74)
    print(f"TEST AUC BY LOSS ({len(seeds)} shuffles each) — {name}")
    print("=" * 74)
    print(summ.round(4).to_string())
    print("\n(contrastive0 = the loss's natural magnitude at start; a big value "
          "means\n it dominates the fused loss unless lambda is reduced.)")

    # ---------------- figure ----------------
    order = list(summ.index)
    cmap = plt.get_cmap("tab10")
    colors = {L: cmap(i % 10) for i, L in enumerate(order)}
    fig, axes = plt.subplots(2, 3, figsize=(18, 9))

    # (a) headline: test AUC per loss
    x = np.arange(len(order))
    axes[0, 0].bar(x, summ.test_auc_mean, yerr=summ.test_auc_std, capsize=4,
                   color=[colors[L] for L in order], edgecolor="white")
    axes[0, 0].axhline(0.5, ls=":", c="grey", lw=1.2, label="random (0.5)")
    axes[0, 0].set_xticks(x); axes[0, 0].set_xticklabels(order, rotation=25,
                                                         ha="right", fontsize=8)
    lo = max(0.0, min(0.48, (summ.test_auc_mean - summ.test_auc_std).min() - 0.03))
    axes[0, 0].set_ylim(lo, min(1.0, (summ.test_auc_mean + summ.test_auc_std).max() + 0.03))
    axes[0, 0].set_title(f"TEST AUC by contrastive loss\n(mean $\\pm$ std over "
                         f"{len(seeds)} shuffles)")
    axes[0, 0].set_ylabel("test AUC")

    def per_loss(ax, col, title, ylab):
        for L in order:
            g = df[df.loss == L].groupby("epoch")[col].mean()
            ax.plot(g.index, g.values, label=L, color=colors[L], lw=1.4)
        ax.set_title(title); ax.set_xlabel("epoch"); ax.set_ylabel(ylab)

    per_loss(axes[0, 1], "val_auc", "Validation AUC per epoch (mean over shuffles)", "AUC")
    per_loss(axes[0, 2], "train_auc", "Train AUC per epoch (mean over shuffles)", "AUC")
    per_loss(axes[1, 0], "train_loss", "TASK loss (train) per epoch", "cross-entropy")
    per_loss(axes[1, 1], "batch_contrastive_loss",
             "CONTRASTIVE loss (raw) per epoch\n(note the different natural scales)",
             "loss")

    # (f) how much of the fused loss the contrastive term takes, per loss
    shares = []
    for L in order:
        s = df[df.loss == L]
        shares.append(s.weighted_contrastive.mean() /
                      max(s.weighted_contrastive.mean() + s.batch_task_loss.mean(), 1e-9))
    axes[1, 2].bar(x, shares, color=[colors[L] for L in order], edgecolor="white")
    axes[1, 2].axhline(0.5, ls="--", c="tab:red", lw=1.3, label="equal contribution")
    axes[1, 2].set_xticks(x); axes[1, 2].set_xticklabels(order, rotation=25,
                                                         ha="right", fontsize=8)
    axes[1, 2].set_ylim(0, 1)
    axes[1, 2].set_title("Contrastive share of the fused loss\n"
                         "(above the red line = task loss is subdued)")
    axes[1, 2].set_ylabel("fraction of total loss")

    for a in axes.ravel():
        a.grid(alpha=0.3); a.legend(fontsize=7)
    fig.suptitle(f"Deep TKCE-joint on {name} — {len(order)} losses x {len(seeds)} "
                 f"shuffles | encoder {args.enc_depth}x{args.enc_width}, "
                 f"{args.head} n_blocks={args.n_blocks}, lr={args.lr}, "
                 f"{args.epochs} epochs"
                 f"{' [balanced]' if args.balance_losses else ''}",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    png = os.path.join(args.out, f"deep_joint_{name}.png")
    fig.savefig(png, dpi=150, bbox_inches="tight")
    print(f"\n[deep] figure -> {png}")
    print(f"[deep] per-epoch data -> {args.csv}/deep_joint_{name}_epochs.csv")
    print(f"[deep] test summary  -> {args.csv}/deep_joint_{name}_test.csv")


if __name__ == "__main__":
    main()
