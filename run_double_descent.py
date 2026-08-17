"""Epoch-wise DOUBLE DESCENT study on one dataset (default: credit, task 361055).

Double descent (Nakkiran et al. 2019, "Deep Double Descent") is the phenomenon
where test error goes DOWN, then UP (classic overfitting), then DOWN AGAIN if you
keep training far past the point where the training loss hits ~zero. To see it you
need three things this script sets up deliberately:

  * a slow, plain optimizer      -> vanilla SGD (no momentum), tiny lr, small batch
  * a very long run              -> thousands of epochs, past interpolation
  * NO regularization damping    -> dropout / weight-decay / L1 all default to 0,
                                    because they suppress the second descent

It trains the three view configurations (x / x+tree / x+tree+deep) and records,
FOR EVERY EPOCH and every model: train + validation + test loss, AUC and accuracy.
(Test is tracked only to plot the phenomenon — model selection never uses it.)

History is flushed to CSV during training, so a disconnect never costs everything.

Outputs (per --out dir):
  dd_<dataset>_epochs.csv        every metric, every epoch, every model
  dd_<dataset>_<model>.png       6-panel per-model diagnostic (log-x)
  dd_<dataset>_compare.png       all three models overlaid
  dd_<dataset>.json              summary: best-val vs final-epoch, tree ceiling

Colab (GPU):
  python -u run_double_descent.py --views 'x' --epochs 4000
  python -u run_double_descent.py --views 'x+tree' --epochs 4000
  python -u run_double_descent.py --views 'x+tree+deep' --epochs 4000
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from run_fusion import (FusionModel, evaluate, fit_encoder_rf,
                        oob_honest_encoding, split_direction_encoding)
from tkce.baselines import fit_tree_baseline
from tkce.data import load_task
from tkce.train import resolve_device

LABELS = {"x": "raw (x only)", "x+tree": "x + tree",
          "x+tree+deep": "FULL (x+tree+deep)"}


# --------------------------------------------------------------------------- #
# checkpointing (survives a Colab disconnect; keeps only the newest file)
# --------------------------------------------------------------------------- #
def _ckpt_list(ckpt_dir, ds_name, key):
    """Existing checkpoints for this (dataset, view-config), oldest -> newest."""
    pat = os.path.join(ckpt_dir, f"ckpt_{ds_name}_{key}_ep*.pt")
    found = []
    for p in glob.glob(pat):
        m = re.search(r"_ep(\d+)\.pt$", p)
        if m:
            found.append((int(m.group(1)), p))
    return [p for _, p in sorted(found)]


def save_checkpoint(ckpt_dir, ds_name, key, epoch, model, opt, hist, best, args):
    """Write the new checkpoint FIRST, then delete the older ones — so there is
    always at least one complete file on disk even if we die mid-write."""
    os.makedirs(ckpt_dir, exist_ok=True)
    new = os.path.join(ckpt_dir, f"ckpt_{ds_name}_{key}_ep{epoch:06d}.pt")
    payload = dict(
        epoch=epoch, model=model.state_dict(), optim=opt.state_dict(),
        hist=hist, best=best,
        rng=dict(torch=torch.get_rng_state(),
                 cuda=(torch.cuda.get_rng_state_all()
                       if torch.cuda.is_available() else None),
                 numpy=np.random.get_state()),
        cfg=dict(views=key, task=args.task, optimizer=args.optimizer,
                 momentum=args.momentum, lr=args.lr, batch_size=args.batch_size,
                 epochs=args.epochs, encoding=args.encoding, seed=args.seed))
    torch.save(payload, new)
    # partial history alongside it, so results are readable without resuming
    pd.DataFrame(hist).to_csv(
        os.path.join(ckpt_dir, f"dd_{ds_name}_{key}_partial.csv"), index=False)
    for old in _ckpt_list(ckpt_dir, ds_name, key):
        if os.path.abspath(old) != os.path.abspath(new):
            try:
                os.remove(old)
            except OSError:
                pass
    return new


def load_checkpoint(ckpt_dir, ds_name, key, model, opt, args):
    """Newest readable checkpoint, or None. Falls back to older files if the
    newest was truncated by a disconnect mid-write."""
    for path in reversed(_ckpt_list(ckpt_dir, ds_name, key)):
        try:
            try:
                ck = torch.load(path, map_location="cpu", weights_only=False)
            except TypeError:                       # torch < 2.0 has no weights_only
                ck = torch.load(path, map_location="cpu")
        except Exception as e:                      # noqa: BLE001 - truncated file
            print(f"  [ckpt] {os.path.basename(path)} unreadable "
                  f"({type(e).__name__}) — trying an older one", flush=True)
            continue
        cfg = ck.get("cfg", {})
        for k in ("task", "optimizer", "lr", "batch_size", "encoding"):
            if k in cfg and cfg[k] != getattr(args, k):
                print(f"  [ckpt] WARNING: checkpoint {k}={cfg[k]} but this run has "
                      f"{getattr(args, k)} — resuming anyway", flush=True)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optim"])
        try:
            torch.set_rng_state(ck["rng"]["torch"])
            if ck["rng"]["cuda"] is not None and torch.cuda.is_available():
                torch.cuda.set_rng_state_all(ck["rng"]["cuda"])
            np.random.set_state(ck["rng"]["numpy"])
        except Exception as e:                      # noqa: BLE001
            print(f"  [ckpt] could not restore RNG ({type(e).__name__}); "
                  f"continuing with a fresh shuffle order", flush=True)
        print(f"  [ckpt] RESUMED from {os.path.basename(path)} "
              f"(epoch {ck['epoch']})", flush=True)
        return ck
    return None


def train_double_descent(views, ds, enc, n_classes, args, device, label, csv_path):
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    model = FusionModel(ds.n_features, enc["train"].shape[1], n_classes,
                        views, args).to(device)
    n_par = sum(p.numel() for p in model.parameters())

    if args.optimizer == "sgd":
        opt = torch.optim.SGD(model.parameters(), lr=args.lr,
                              momentum=args.momentum,
                              weight_decay=args.weight_decay)
        opt_desc = (f"vanilla SGD (momentum={args.momentum:g})"
                    if args.momentum == 0 else f"SGD momentum={args.momentum:g}")
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                weight_decay=args.weight_decay)
        opt_desc = "AdamW"

    n_steps = int(np.ceil(len(ds.y_train) / args.batch_size))
    print(f"\n{'=' * 70}\n[{label}] views={views}  concat_dim={model.cat_dim}  "
          f"params={n_par/1e6:.2f}M\n"
          f"  optimizer={opt_desc}  lr={args.lr:g}  batch={args.batch_size}  "
          f"({n_steps} steps/epoch, {n_steps*args.epochs/1e6:.2f}M steps total)\n"
          f"  dropout={args.dropout:g} weight_decay={args.weight_decay:g} "
          f"l1={args.l1:g}  <- keep at 0 to let double descent appear\n{'=' * 70}",
          flush=True)

    loader = DataLoader(
        TensorDataset(torch.from_numpy(ds.X_train).to(device),
                      torch.from_numpy(enc["train"]).to(device),
                      torch.from_numpy(ds.y_train).to(device)),
        batch_size=args.batch_size, shuffle=True)

    # ---- resume from Drive if a checkpoint is there ----
    key = "+".join(views)
    hist, best, start_epoch = [], {"val_auc": -1.0, "epoch": 0}, 1
    if args.ckpt_dir and not args.fresh:
        ck = load_checkpoint(args.ckpt_dir, ds.name, key, model, opt, args)
        if ck is not None:
            hist, best = ck["hist"], ck["best"]
            start_epoch = ck["epoch"] + 1
            if start_epoch > args.epochs:
                print(f"  [ckpt] already finished ({ck['epoch']} epochs) — "
                      f"skipping training, rebuilding outputs", flush=True)
    if args.ckpt_dir:
        print(f"  [ckpt] dir={args.ckpt_dir}  every {args.ckpt_every} epochs "
              f"(only the newest file is kept)", flush=True)
    if 1 < start_epoch <= args.epochs:
        print(f"  [ckpt] continuing at epoch {start_epoch}/{args.epochs}", flush=True)

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        for xb, tb, yb in loader:
            opt.zero_grad()
            loss = F.cross_entropy(model(xb, tb), yb)
            if args.l1 > 0:
                loss = loss + args.l1 * sum(p.abs().sum()
                                            for p in model.parameters() if p.ndim >= 2)
            loss.backward(); opt.step()

        if epoch % args.eval_every and epoch != 1 and epoch != args.epochs:
            continue                                   # skip eval on this epoch

        tr = evaluate(model, ds.X_train, enc["train"], ds.y_train, device)
        va = evaluate(model, ds.X_val, enc["val"], ds.y_val, device)
        te = evaluate(model, ds.X_test, enc["test"], ds.y_test, device)
        hist.append(dict(model=label, views="+".join(views), epoch=epoch,
                         train_loss=tr["loss"], val_loss=va["loss"], test_loss=te["loss"],
                         train_auc=tr["auc"], val_auc=va["auc"], test_auc=te["auc"],
                         train_acc=tr["accuracy"], val_acc=va["accuracy"],
                         test_acc=te["accuracy"]))
        if va["auc"] > best["val_auc"]:
            best = {"val_auc": va["auc"], "epoch": epoch,
                    "test_auc": te["auc"], "test_acc": te["accuracy"]}

        if epoch % args.log_every == 0 or epoch == 1 or epoch == args.epochs:
            print(f"  ep {epoch:5d}/{args.epochs}  "
                  f"train_loss={tr['loss']:.4f} val_loss={va['loss']:.4f} "
                  f"test_loss={te['loss']:.4f} | "
                  f"train_auc={tr['auc']:.4f} val_auc={va['auc']:.4f} "
                  f"test_auc={te['auc']:.4f}", flush=True)
        if epoch % args.flush_every == 0:              # crash insurance (local)
            pd.DataFrame(hist).to_csv(csv_path, index=False)
        if args.ckpt_dir and epoch % args.ckpt_every == 0:
            save_checkpoint(args.ckpt_dir, ds.name, key, epoch, model, opt,
                            hist, best, args)

    pd.DataFrame(hist).to_csv(csv_path, index=False)
    if args.ckpt_dir and hist:                        # final checkpoint
        save_checkpoint(args.ckpt_dir, ds.name, key, hist[-1]["epoch"], model,
                        opt, hist, best, args)
    last = hist[-1]
    print(f"  -> FINAL epoch {last['epoch']}: test_auc={last['test_auc']:.4f}  "
          f"train_loss={last['train_loss']:.5f}\n"
          f"  -> BEST-VAL epoch {best['epoch']}: test_auc={best['test_auc']:.4f} "
          f"(val {best['val_auc']:.4f})", flush=True)
    return hist, dict(model=label, views="+".join(views), params_M=round(n_par/1e6, 2),
                      concat_dim=model.cat_dim,
                      final_epoch=last["epoch"], final_test_auc=last["test_auc"],
                      final_train_loss=last["train_loss"],
                      final_val_loss=last["val_loss"], final_test_loss=last["test_loss"],
                      best_val_epoch=best["epoch"], best_val_auc=best["val_auc"],
                      best_val_test_auc=best["test_auc"])


# --------------------------------------------------------------------------- #
# figures
# --------------------------------------------------------------------------- #
def per_model_figure(df, label, ceiling, path, logx=True):
    g = df[df.model == label].sort_values("epoch")
    fig, ax = plt.subplots(2, 3, figsize=(19, 10))

    def setx(a):
        if logx:
            a.set_xscale("log")
        a.set_xlabel("epoch"); a.grid(alpha=.3)

    ax[0, 0].plot(g.epoch, g.train_loss, label="train", color="tab:blue")
    ax[0, 0].plot(g.epoch, g.val_loss, label="validation", color="tab:orange")
    ax[0, 0].plot(g.epoch, g.test_loss, label="test", color="tab:red")
    ax[0, 0].set_title("Loss vs epoch\n(double descent = test dips, rises, dips again)")
    ax[0, 0].set_ylabel("cross-entropy"); setx(ax[0, 0]); ax[0, 0].legend(fontsize=8)

    ax[0, 1].plot(g.epoch, g.train_auc, label="train", color="tab:blue")
    ax[0, 1].plot(g.epoch, g.val_auc, label="validation", color="tab:orange")
    ax[0, 1].plot(g.epoch, g.test_auc, label="test", color="tab:red")
    if ceiling is not None:
        ax[0, 1].axhline(ceiling, ls="--", c="green", lw=1, label=f"tree {ceiling:.3f}")
    ax[0, 1].set_title("AUC vs epoch"); ax[0, 1].set_ylabel("AUC")
    setx(ax[0, 1]); ax[0, 1].legend(fontsize=8)

    # the money panel: test loss with its running minimum + the bump
    ax[0, 2].plot(g.epoch, g.test_loss, color="tab:red", label="test loss")
    ax[0, 2].plot(g.epoch, g.val_loss, color="tab:orange", alpha=.6, label="val loss")
    imin = g.test_loss.idxmin()
    ax[0, 2].scatter([g.loc[imin, "epoch"]], [g.loc[imin, "test_loss"]], zorder=5,
                     color="black", s=45,
                     label=f"min @ ep {int(g.loc[imin,'epoch'])}")
    ax[0, 2].set_title("Test/val loss — where is the bump?")
    ax[0, 2].set_ylabel("cross-entropy"); setx(ax[0, 2]); ax[0, 2].legend(fontsize=8)

    ax[1, 0].plot(g.epoch, g.train_loss, color="tab:blue")
    ax[1, 0].set_yscale("log"); ax[1, 0].set_title(
        "Train loss (log y)\nreaching ~0 = the interpolation point")
    ax[1, 0].set_ylabel("cross-entropy (log)"); setx(ax[1, 0])

    ax[1, 1].plot(g.epoch, g.train_auc - g.test_auc, color="tab:purple")
    ax[1, 1].axhline(0, ls=":", c="grey")
    ax[1, 1].set_title("Generalization gap (train AUC − test AUC)")
    ax[1, 1].set_ylabel("gap"); setx(ax[1, 1])

    ax[1, 2].plot(g.epoch, g.train_acc, label="train", color="tab:blue")
    ax[1, 2].plot(g.epoch, g.val_acc, label="validation", color="tab:orange")
    ax[1, 2].plot(g.epoch, g.test_acc, label="test", color="tab:red")
    ax[1, 2].set_title("Accuracy vs epoch"); ax[1, 2].set_ylabel("accuracy")
    setx(ax[1, 2]); ax[1, 2].legend(fontsize=8)

    fig.suptitle(f"Epoch-wise double descent — {label}", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def compare_figure(df, ceiling, path, logx=True):
    models = list(dict.fromkeys(df.model))
    cmap = plt.get_cmap("tab10")
    col = {m: cmap(i % 10) for i, m in enumerate(models)}
    fig, ax = plt.subplots(2, 2, figsize=(15, 10))

    def panel(a, ycol, title, ylab, hline=None, logy=False):
        for m in models:
            g = df[df.model == m].sort_values("epoch")
            a.plot(g.epoch, g[ycol], color=col[m], lw=1.4, label=m)
        if hline is not None:
            a.axhline(hline, ls="--", c="green", lw=1, label="tree ceiling")
        if logx:
            a.set_xscale("log")
        if logy:
            a.set_yscale("log")
        a.set_title(title); a.set_xlabel("epoch"); a.set_ylabel(ylab)
        a.grid(alpha=.3); a.legend(fontsize=8)

    panel(ax[0, 0], "test_loss", "TEST loss vs epoch", "cross-entropy")
    panel(ax[0, 1], "test_auc", "TEST AUC vs epoch", "AUC", hline=ceiling)
    panel(ax[1, 0], "train_loss", "TRAIN loss vs epoch (log y)", "cross-entropy", logy=True)
    panel(ax[1, 1], "val_loss", "VALIDATION loss vs epoch", "cross-entropy")
    fig.suptitle("Double descent — all three view configurations",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", type=int, default=361055)          # credit
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-rows", type=int, default=16000)
    ap.add_argument("--views", default="x;x+tree;x+tree+deep",
                    help="semicolon-separated view configs to run")
    # ---- the double-descent recipe ----
    ap.add_argument("--optimizer", default="sgd", choices=["sgd", "adamw"])
    ap.add_argument("--momentum", type=float, default=0.0,
                    help="0 = vanilla SGD (the default for this study)")
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=4000)
    ap.add_argument("--dropout", type=float, default=0.0,
                    help="0 for a clean study — regularization suppresses double descent")
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--l1", type=float, default=0.0)
    # ---- encoding ----
    ap.add_argument("--encoding", default="oob", choices=["infold", "oob"])
    ap.add_argument("--tau", type=float, default=0.0)
    ap.add_argument("--rf-trees", type=int, default=100)
    ap.add_argument("--rf-depth", type=int, default=6)
    ap.add_argument("--rf-min-leaf", type=int, default=5)
    # ---- architecture (same as run_fusion) ----
    ap.add_argument("--fusion", default="tabresnet", choices=["tabresnet", "mlp"])
    ap.add_argument("--feat-dim", type=int, default=128)
    ap.add_argument("--ext-d", type=int, default=192)
    ap.add_argument("--ext-d-hidden", type=int, default=256)
    ap.add_argument("--ext-blocks", type=int, default=3)
    ap.add_argument("--d", type=int, default=256)
    ap.add_argument("--d-hidden", type=int, default=512)
    ap.add_argument("--n-blocks", type=int, default=4)
    # ---- bookkeeping ----
    ap.add_argument("--eval-every", type=int, default=1,
                    help="evaluate every N epochs (1 = every epoch, as requested)")
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--flush-every", type=int, default=100,
                    help="write the CSV every N epochs so a crash keeps the history")
    ap.add_argument("--linear-x", action="store_true",
                    help="plot epochs on a linear axis (default is log)")
    # ---- checkpoint / resume (point this at Google Drive on Colab) ----
    ap.add_argument("--ckpt-dir", default="",
                    help="save a resumable checkpoint here (e.g. a mounted "
                         "Google Drive folder). Empty = no checkpointing")
    ap.add_argument("--ckpt-every", type=int, default=10,
                    help="checkpoint every N epochs; only the newest is kept")
    ap.add_argument("--fresh", action="store_true",
                    help="ignore existing checkpoints and start from epoch 1")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", default="results/double_descent")
    args = ap.parse_args()

    device = resolve_device(args.device)
    os.makedirs(args.out, exist_ok=True)
    ds = load_task(args.task, seed=args.seed, max_rows=args.max_rows)
    if ds.task_type != "classification":
        raise SystemExit(f"{ds.name} is not classification.")
    C = ds.n_classes
    print(f"\n=== DOUBLE DESCENT: {ds.name} (task {args.task}) | "
          f"{ds.n_features} feats, {C} classes | device={device} ===")
    print(f"[data] train={len(ds.y_train)} val={len(ds.y_val)} test={len(ds.y_test)}",
          flush=True)

    # tree encoding (view 2)
    print(f"[view2] fitting encoding RF ({args.rf_trees} trees, depth {args.rf_depth}) ...",
          flush=True)
    rf = fit_encoder_rf(ds.X_train, ds.y_train, args)
    enc = {"train": split_direction_encoding(rf, ds.X_train, args.tau),
           "val":   split_direction_encoding(rf, ds.X_val, args.tau),
           "test":  split_direction_encoding(rf, ds.X_test, args.tau)}
    print(f"[view2] tree-encoding width = {enc['train'].shape[1]} split bits", flush=True)
    if args.encoding == "oob":
        enc["train"], mean_oob = oob_honest_encoding(rf, enc["train"], len(ds.y_train))
        print(f"[view2] OOB-honest: train rows keep bits from {mean_oob:.1f}/"
              f"{args.rf_trees} trees on average", flush=True)

    # tree ceiling for reference lines
    print("[ceiling] fitting tree baselines ...", flush=True)
    ceil = {}
    for name in ("xgboost", "lightgbm", "random_forest"):
        te, _, _ = fit_tree_baseline(name, ds, {"seed": args.seed})
        ceil[name] = te["auc"]
        print(f"  {name:14s} AUC={te['auc']:.4f}", flush=True)
    tree_ceiling = max(ceil.values())

    # ---- run each view config ----
    run_views = [v.strip().split("+") for v in args.views.split(";") if v.strip()]
    all_hist, summaries = [], []
    for v in run_views:
        key = "+".join(v)
        label = LABELS.get(key, key)
        csv_path = os.path.join(args.out, f"dd_{ds.name}_{key.replace('+','-')}.csv")
        hist, summ = train_double_descent(v, ds, enc, C, args, device, label, csv_path)
        all_hist.extend(hist); summaries.append(summ)
        per_model_figure(pd.DataFrame(hist), label, tree_ceiling,
                         os.path.join(args.out, f"dd_{ds.name}_{key.replace('+','-')}.png"),
                         logx=not args.linear_x)

    df = pd.DataFrame(all_hist)
    df.to_csv(os.path.join(args.out, f"dd_{ds.name}_epochs.csv"), index=False)
    if len(run_views) > 1:
        compare_figure(df, tree_ceiling,
                       os.path.join(args.out, f"dd_{ds.name}_compare.png"),
                       logx=not args.linear_x)

    # ---- summary ----
    print("\n" + "=" * 78)
    print(f"DOUBLE DESCENT SUMMARY — {ds.name} | {args.optimizer} lr={args.lr:g} "
          f"batch={args.batch_size} epochs={args.epochs}")
    print("=" * 78)
    print(f"  tree ceiling: {tree_ceiling:.4f}")
    print(f"  {'model':22s} {'final test':>11s} {'best-val test':>14s} "
          f"{'min test loss':>14s} {'@epoch':>7s}")
    for s, v in zip(summaries, run_views):
        g = df[df.views == "+".join(v)]
        i = g.test_loss.idxmin()
        print(f"  {s['model']:22s} {s['final_test_auc']:11.4f} "
              f"{s['best_val_test_auc']:14.4f} {g.loc[i,'test_loss']:14.4f} "
              f"{int(g.loc[i,'epoch']):7d}")
    print("\n  Double descent shows up as: test loss falls -> RISES -> falls again,")
    print("  usually after the train loss has essentially reached zero.")

    with open(os.path.join(args.out, f"dd_{ds.name}.json"), "w") as f:
        json.dump(dict(dataset=ds.name, task=args.task, optimizer=args.optimizer,
                       momentum=args.momentum, lr=args.lr, batch_size=args.batch_size,
                       epochs=args.epochs, dropout=args.dropout,
                       weight_decay=args.weight_decay, l1=args.l1,
                       encoding=args.encoding, tree_ceiling=tree_ceiling,
                       ceiling=ceil, results=summaries), f, indent=2)
    print(f"\n[dd] epoch CSV -> {args.out}/dd_{ds.name}_epochs.csv")
    print(f"[dd] figures   -> {args.out}/dd_{ds.name}_*.png")


if __name__ == "__main__":
    main()
