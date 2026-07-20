"""Deep TKCE-joint study: sweep LOSSES and/or LAMBDA over multiple SHUFFLES.

Deep Siamese encoder + deep TabResNet, many epochs, slow learning rate, NO early
stopping. Every run is repeated over several random train/val/test shuffles.

Two sweeps (pick one by giving several values):
  * --losses  : compare contrastive losses at a fixed lambda
  * --lambdas : sweep the fused-loss weight for one loss  <-- "sweep through!"

Why the lambda sweep matters: the contrastive (kernel-embedding) loss is
numerically much larger than the task cross-entropy (~5.2 vs ~0.77 here), so at
lambda=0.5 it takes ~78% of the fused loss and the task loss is subdued. Treating
the contrastive term as a *regularizer* means it should contribute only ~10%,
i.e. lambda ~ 0.015 on this data. The sweep finds that empirically.

Tracks per epoch, per run: task loss (train+val), contrastive loss, AUC
(train+val), accuracy (train+val), the effective weights and the contrastive
share of the fused loss.

NOTE: the joint regime supports the in-batch anchor/positive losses only
(infonce, aninfonce, clip_infonce, contrastive, triplet). supcon and
kernel_regression are two-stage only and are skipped with a message.

Examples:
  # lambda sweep (the professor's request)
  python run_deep_joint.py --losses infonce --lambdas 0,0.005,0.015,0.05,0.15,0.5 \
      --seeds 0,1,2 --epochs 600 --lr 1e-6

  # loss comparison at a fixed lambda
  python run_deep_joint.py --lambdas 0.015 --seeds 0,1,2 --epochs 600 --lr 1e-6
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
from tkce.losses import UncertaintyWeighting, apply_pair_loss, build_contrastive
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


def run_one(loss_name, lam, seed, args, device):
    """One (contrastive loss, lambda, shuffle) run."""
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
    # the contrastive pairs get their OWN (larger) batch so InfoNCE keeps plenty
    # of in-batch negatives even when the supervised batch is small
    cbs = args.contrastive_batch_size or args.batch_size
    ap_gen = None
    if len(ap_ds) > 0 and lam > 0:
        ap_gen = _cycle(DataLoader(ap_ds, batch_size=cbs,
                                   shuffle=True, drop_last=False))

    uw = UncertaintyWeighting(2).to(device) if args.uncertainty_weighting else None
    params = list(model.parameters()) + list(contrast.parameters())
    if uw is not None:
        params += list(uw.parameters())
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    sup_loader = DataLoader(
        TensorDataset(torch.from_numpy(ds.X_train).to(device),
                      torch.from_numpy(ds.y_train).to(device)),
        batch_size=args.batch_size, shuffle=True)

    # starting magnitudes -> optional auto-balance
    lam_eff, t0, c0 = lam, float("nan"), float("nan")
    with torch.no_grad():
        xb0, yb0 = next(iter(sup_loader))
        t0 = F.cross_entropy(model(xb0)[0], yb0).item()
        if ap_gen is not None:
            xi0, xj0 = next(ap_gen)
            c0 = apply_pair_loss(loss_name, contrast,
                                 enc(xi0.to(device)), enc(xj0.to(device))).item()
    if args.balance_losses and np.isfinite(c0) and c0 > 1e-8:
        lam_eff = lam * (t0 / c0)
    mode = ("uncertainty (learned)" if uw is not None else
            "balanced" if args.balance_losses else "fixed")
    print(f"  [{loss_name} | lam={lam:g} | seed {seed}] task0={t0:.3f} "
          f"contrastive0={c0:.3f} (ratio {c0/max(t0,1e-8):.1f}x) "
          f"weighting={mode} lambda_eff={lam_eff:.5f} | "
          f"batch task={args.batch_size} contrastive={cbs} "
          f"({2*cbs-2} in-batch negatives)", flush=True)

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
                c_val = c_loss.item()
                loss = (uw([t_loss, c_loss]) if uw is not None
                        else t_loss + lam_eff * c_loss)
            loss.backward(); opt.step()
            task_sum += t_loss.item(); con_sum += c_val; nb += 1

        if uw is not None:
            w_task, w_con = (float(v) for v in uw.weights())
        else:
            w_task, w_con = 1.0, lam_eff

        tr_loss, tr_auc, tr_acc = _eval_split(model, ds.X_train, ds.y_train, device)
        va_loss, va_auc, va_acc = _eval_split(model, ds.X_val, ds.y_val, device)
        wt, wc = w_task * task_sum / nb, w_con * con_sum / nb
        hist.append(dict(loss=loss_name, lam=lam, seed=seed, epoch=epoch,
                         lam_eff=lam_eff, w_task=w_task, w_contrastive=w_con,
                         batch_task_loss=task_sum / nb,
                         batch_contrastive_loss=con_sum / nb,
                         weighted_task=wt, weighted_contrastive=wc,
                         contrastive_share=wc / max(wt + wc, 1e-9),
                         train_loss=tr_loss, val_loss=va_loss,
                         train_auc=tr_auc, val_auc=va_auc,
                         train_acc=tr_acc, val_acc=va_acc))
        if epoch % max(1, args.epochs // 6) == 0:
            print(f"    epoch {epoch:4d}/{args.epochs} train_auc={tr_auc:.4f} "
                  f"val_auc={va_auc:.4f}", flush=True)

    te_loss, te_auc, te_acc = _eval_split(model, ds.X_test, ds.y_test, device)
    h = pd.DataFrame(hist)
    best = h.loc[h.val_auc.idxmax()]
    test = dict(loss=loss_name, lam=lam, seed=seed, dataset=ds.name,
                test_auc=te_auc, test_acc=te_acc, test_loss=te_loss,
                train_auc_final=h.train_auc.iloc[-1],
                best_val_auc=best.val_auc, best_val_epoch=int(best.epoch),
                lam_eff=lam_eff, contrastive0=c0, task0=t0,
                final_contrastive_share=h.contrastive_share.iloc[-1])
    print(f"  -> TEST auc={te_auc:.4f} acc={te_acc:.4f} "
          f"(train_auc_final={h.train_auc.iloc[-1]:.4f})\n", flush=True)
    return h, test, ds.name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", type=int, default=361070)
    ap.add_argument("--seeds", default="0,1,2", help="shuffles (new split each)")
    ap.add_argument("--losses", default=",".join(JOINT_LOSSES))
    ap.add_argument("--lambdas", default="0.5",
                    help="fused-loss weights to sweep, e.g. 0,0.005,0.015,0.05,0.15,0.5")
    ap.add_argument("--epochs", type=int, default=600)
    ap.add_argument("--lr", type=float, default=1e-6,
                    help="slow LR; raise only if training is too flat")
    ap.add_argument("--batch-size", type=int, default=64,
                    help="batch for the TASK loss; small = more updates + more "
                         "gradient noise (regularisation) on a small dataset")
    ap.add_argument("--contrastive-batch-size", type=int, default=None,
                    help="separate batch for the CONTRASTIVE pairs. InfoNCE learns "
                         "from in-batch negatives, so keep this LARGE (e.g. 256) "
                         "even when --batch-size is small. Defaults to --batch-size.")
    ap.add_argument("--head", default="tabresnet", choices=["tabresnet", "mlp"])
    ap.add_argument("--enc-width", type=int, default=512)
    ap.add_argument("--enc-depth", type=int, default=6)
    ap.add_argument("--embedding-dim", type=int, default=128)
    ap.add_argument("--enc-dropout", type=float, default=0.1)
    ap.add_argument("--d", type=int, default=256)
    ap.add_argument("--d-hidden", type=int, default=512)
    ap.add_argument("--n-blocks", type=int, default=8)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--balance-losses", action="store_true")
    ap.add_argument("--uncertainty-weighting", action="store_true")
    ap.add_argument("--temperature", type=float, default=0.1)
    ap.add_argument("--margin", type=float, default=1.0)
    ap.add_argument("--pos-threshold", type=float, default=0.6)
    ap.add_argument("--k-n-estimators", type=int, default=200)
    ap.add_argument("--k-max-depth", type=int, default=4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--max-rows", type=int, default=16000)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", default="paper/figures")
    ap.add_argument("--csv", default="results/analysis")
    ap.add_argument("--tag", default="", help="suffix for output filenames")
    args = ap.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    losses = [l.strip() for l in args.losses.split(",")]
    lambdas = [float(x) for x in args.lambdas.split(",")]
    sweep_lambda = len(lambdas) > 1
    device = resolve_device(args.device)
    os.makedirs(args.out, exist_ok=True); os.makedirs(args.csv, exist_ok=True)

    print(f"[deep] {len(losses)} loss(es) x {len(lambdas)} lambda(s) x {len(seeds)} "
          f"shuffles = {len(losses)*len(lambdas)*len(seeds)} runs")
    cbs = args.contrastive_batch_size or args.batch_size
    print(f"[deep] epochs={args.epochs}  lr={args.lr:g}  "
          f"BATCH(task)={args.batch_size}  BATCH(contrastive)={cbs}  "
          f"encoder={args.enc_depth}x{args.enc_width}  {args.head}"
          f"(n_blocks={args.n_blocks})  device={device}\n", flush=True)

    hists, tests, name = [], [], None
    for L in losses:
        for lam in lambdas:
            print(f"===== loss={L}  lambda={lam:g} =====", flush=True)
            for s in seeds:
                try:
                    h, t, name = run_one(L, lam, s, args, device)
                    hists.append(h); tests.append(t)
                except ValueError as e:
                    print(f"  SKIP {L}: {e}\n", flush=True); break

    if not tests:
        raise SystemExit("no runs completed")
    df = pd.concat(hists, ignore_index=True)
    tdf = pd.DataFrame(tests)
    # label each configuration
    key = "lam" if sweep_lambda and len(losses) == 1 else (
          "loss" if len(lambdas) == 1 else "variant")
    if key == "variant":
        df["variant"] = df.loss + " λ=" + df.lam.map(lambda v: f"{v:g}")
        tdf["variant"] = tdf.loss + " λ=" + tdf.lam.map(lambda v: f"{v:g}")
    tag = f"_{args.tag}" if args.tag else ""
    df.to_csv(os.path.join(args.csv, f"deep_joint_{name}{tag}_epochs.csv"), index=False)
    tdf.to_csv(os.path.join(args.csv, f"deep_joint_{name}{tag}_test.csv"), index=False)

    # ---------------- summary ----------------
    summ = (tdf.groupby(key)
              .agg(test_auc_mean=("test_auc", "mean"), test_auc_std=("test_auc", "std"),
                   test_acc_mean=("test_acc", "mean"),
                   train_auc_final=("train_auc_final", "mean"),
                   n=("test_auc", "size"), lam_eff=("lam_eff", "mean"),
                   contrastive_share=("final_contrastive_share", "mean")))
    summ["test_auc_std"] = summ["test_auc_std"].fillna(0.0)
    if key != "lam":
        summ = summ.sort_values("test_auc_mean", ascending=False)
    print("=" * 92)
    print(f"RESULTS ({len(seeds)} shuffles each) — {name} | batch={args.batch_size} "
          f"lr={args.lr:g} epochs={args.epochs}")
    print("=" * 92)
    print(summ.round(4).to_string())
    if sweep_lambda:
        best = summ.test_auc_mean.idxmax()
        print(f"\nBest lambda = {best:g}  (test AUC {summ.test_auc_mean.max():.4f}, "
              f"contrastive share {summ.loc[best,'contrastive_share']:.0%})")
    print("\ntrain_auc_final near 1.0 => the model memorised the training set; "
          "lower the LR or add regularisation.")

    # ---------------- figure ----------------
    order = list(summ.index)
    cmap = plt.get_cmap("tab10")
    colors = {v: cmap(i % 10) for i, v in enumerate(order)}
    lbl = (lambda v: f"λ={v:g}") if key == "lam" else (lambda v: str(v))
    fig, axes = plt.subplots(3, 3, figsize=(18, 13))

    # (a) headline
    if key == "lam":
        xs = [max(v, 1e-4) for v in order]           # lambda=0 shown at the left edge
        axes[0, 0].errorbar(xs, summ.test_auc_mean, yerr=summ.test_auc_std,
                            marker="o", capsize=4, color="tab:blue")
        axes[0, 0].set_xscale("log"); axes[0, 0].set_xlabel("λ (contrastive weight)")
        axes[0, 0].set_title("TEST AUC vs λ  (mean ± std over shuffles)")
    else:
        x = np.arange(len(order))
        axes[0, 0].bar(x, summ.test_auc_mean, yerr=summ.test_auc_std, capsize=4,
                       color=[colors[v] for v in order], edgecolor="white")
        axes[0, 0].set_xticks(x)
        axes[0, 0].set_xticklabels([lbl(v) for v in order], rotation=25,
                                   ha="right", fontsize=8)
        axes[0, 0].set_title("TEST AUC (mean ± std over shuffles)")
    axes[0, 0].axhline(0.5, ls=":", c="grey", lw=1.2, label="random (0.5)")
    axes[0, 0].set_ylabel("test AUC")

    def per_variant(ax, col, title, ylab):
        for v in order:
            g = df[df[key] == v].groupby("epoch")[col].mean()
            ax.plot(g.index, g.values, label=lbl(v), color=colors[v], lw=1.4)
        ax.set_title(title); ax.set_xlabel("epoch"); ax.set_ylabel(ylab)

    per_variant(axes[0, 1], "train_loss", "TASK loss — TRAIN", "cross-entropy")
    per_variant(axes[0, 2], "val_loss", "TASK loss — VALIDATION", "cross-entropy")
    per_variant(axes[1, 0], "batch_contrastive_loss",
                "CONTRASTIVE loss (raw)", "loss")
    per_variant(axes[1, 1], "train_auc", "AUC — TRAIN", "AUC")
    per_variant(axes[1, 2], "val_auc", "AUC — VALIDATION", "AUC")
    per_variant(axes[2, 0], "train_acc", "Accuracy — TRAIN", "accuracy")
    per_variant(axes[2, 1], "val_acc", "Accuracy — VALIDATION", "accuracy")
    per_variant(axes[2, 2], "contrastive_share",
                "Contrastive share of the fused loss", "fraction of total loss")
    axes[2, 2].axhline(0.5, ls="--", c="tab:red", lw=1.2, label="equal")
    axes[2, 2].axhline(0.1, ls=":", c="tab:green", lw=1.2, label="regularizer (10%)")
    axes[2, 2].set_ylim(0, 1)

    for a in axes.ravel():
        a.grid(alpha=0.3); a.legend(fontsize=7)
    fig.suptitle(f"Deep TKCE-joint on {name} — {len(seeds)} shuffles | "
                 f"batch task={args.batch_size}/contrastive={cbs}, "
                 f"lr={args.lr:g}, {args.epochs} epochs | "
                 f"encoder {args.enc_depth}x{args.enc_width}, {args.head} "
                 f"n_blocks={args.n_blocks}"
                 f"{' [balanced]' if args.balance_losses else ''}"
                 f"{' [learned weights]' if args.uncertainty_weighting else ''}",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    png = os.path.join(args.out, f"deep_joint_{name}{tag}.png")
    fig.savefig(png, dpi=150, bbox_inches="tight")
    print(f"\n[deep] figure -> {png}")
    print(f"[deep] data -> {args.csv}/deep_joint_{name}{tag}_epochs.csv "
          f"and _test.csv")


if __name__ == "__main__":
    main()
