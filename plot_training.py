"""Plot loss and AUC curves from a run_fusion.py output directory.

Two levels of detail are available:

  * per-EPOCH  (fusion_<ds>_epochs.csv)  — train/val loss and AUC, always written
  * per-BATCH  (fusion_<ds>_batches.csv) — written when --log-batches is passed;
    every mini-batch's loss, within-epoch running-mean loss, accuracy and AUC

Use from a notebook:

    from plot_training import plot_training_curves, plot_loss, plot_auc, plot_capacity
    plot_training_curves("results/fusion/cap_1_full")          # the 2x2 overview
    plot_loss("results/fusion/cap_1_full")                     # loss only
    plot_auc("results/fusion/cap_1_full")                      # AUC only
    plot_capacity(sorted(glob.glob("results/fusion/cap_*")))   # compare the arms

or from the shell:

    python plot_training.py --run results/fusion/cap_1_full
    python plot_training.py --compare 'results/fusion/cap_*'
"""

from __future__ import annotations

import argparse
import glob
import json
import os

import matplotlib
if not os.environ.get("DISPLAY") and matplotlib.get_backend().lower() != "agg":
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Palette: validated categorical slots (blue / orange / aqua / violet) plus ink
# tokens. Colour carries series identity; every series is also labelled.
BLUE, ORANGE, AQUA, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7"
INK, MUTED, FAINT = "#0b0b0b", "#52514e", "#c8c8c4"

_STYLE = {
    "font.size": 10.5, "axes.grid": True, "grid.alpha": 0.22,
    "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#9a9a96", "figure.facecolor": "white",
    "axes.facecolor": "white", "legend.frameon": False,
}


# --------------------------------------------------------------------------- #
# loading
# --------------------------------------------------------------------------- #
def load_run(run_dir: str):
    """Return (epochs_df, batches_df | None, summary_dict | None)."""
    ep = glob.glob(os.path.join(run_dir, "fusion_*_epochs.csv"))
    ba = glob.glob(os.path.join(run_dir, "fusion_*_batches.csv"))
    js = glob.glob(os.path.join(run_dir, "fusion_*.json"))
    js = [f for f in js if "epochs" not in f and "batches" not in f]
    if not ep:
        raise FileNotFoundError(f"no fusion_*_epochs.csv in {run_dir}")
    edf = pd.read_csv(ep[0])
    bdf = pd.read_csv(ba[0]) if ba else None
    summary = json.load(open(js[0])) if js else None
    return edf, bdf, summary


def _pick_model(df: pd.DataFrame, model: str | None) -> str:
    """Default to the x+tree config when present — that is the model of interest."""
    names = list(dict.fromkeys(df["model"]))
    if model is not None:
        matches = [n for n in names if model.lower() in n.lower()]
        if not matches:
            raise ValueError(f"model {model!r} not in {names}")
        return matches[0]
    for n in names:
        if "tree" in n.lower():
            return n
    return names[0]


def _epoch_mean(edf, name):
    """Average the per-epoch history over ensemble members."""
    return edf[edf["model"] == name].groupby("epoch").mean(numeric_only=True)


def _batch_slice(bdf, name, member):
    b = bdf[bdf["model"] == name]
    if member is not None and "member" in b.columns:
        b = b[b["member"] == member]
    return b.sort_values("step")


def _label(run_dir, summary, name):
    tag = os.path.basename(os.path.normpath(run_dir))
    if summary:
        for r in summary.get("results", []):
            if r.get("model") == name and r.get("params"):
                return f"{tag} · {name} · {r['params']:,} params"
    return f"{tag} · {name}"


# --------------------------------------------------------------------------- #
# per-batch panels
# --------------------------------------------------------------------------- #
def _batch_panel(ax, b, col, ylab, title, smooth=None, max_epoch=None,
                 mark_epochs=True):
    if max_epoch:
        b = b[b["epoch"] <= max_epoch]
    y = b[col].to_numpy(dtype=float)
    x = b["step"].to_numpy()
    ok = ~np.isnan(y)
    n_per_epoch = int(b["batch"].max()) if len(b) else 1
    if smooth is None:
        smooth = max(2, n_per_epoch // 2)          # half an epoch
    ax.plot(x[ok], y[ok], lw=0.6, color=FAINT, label="single mini-batch")
    roll = pd.Series(y).rolling(smooth, min_periods=1).mean().to_numpy()
    ax.plot(x[ok], roll[ok], lw=2, color=BLUE,
            label=f"rolling mean ({smooth} batches)")
    if mark_epochs and len(b):
        n_ep = int(b["epoch"].max())
        step = max(1, n_ep // 20)                  # at most ~20 guides
        for e in range(step, n_ep + 1, step):
            ax.axvline(e * n_per_epoch, color=ORANGE, lw=0.4, alpha=0.5)
        ax.plot([], [], color=ORANGE, lw=0.6, alpha=0.7,
                label=f"epoch boundary (every {step})")
    ax.set_xlabel("mini-batch step")
    ax.set_ylabel(ylab)
    ax.set_title(title, fontweight="bold", loc="left", fontsize=10.5)
    ax.legend(fontsize=8.5, loc="best")


def _epoch_panel(ax, e, cols, ylab, title, logx=True, best_ep=None):
    for (col, color, lab) in cols:
        if col in e.columns:
            ax.plot(e.index, e[col], lw=1.8, color=color, label=lab)
    if best_ep:
        ax.axvline(best_ep, color=VIOLET, lw=1.2, ls=":",
                   label=f"best validation (epoch {best_ep})")
    logx = logx and len(e) >= 20      # log ticks look silly over a handful of epochs
    if logx:
        ax.set_xscale("log")
    ax.set_xlabel("epoch" + (" (log scale)" if logx else ""))
    ax.set_ylabel(ylab)
    ax.set_title(title, fontweight="bold", loc="left", fontsize=10.5)
    ax.legend(fontsize=8.5, loc="best")


# --------------------------------------------------------------------------- #
# public plots
# --------------------------------------------------------------------------- #
def plot_loss(run_dir, model=None, member=0, out=None, smooth=None,
              max_epoch=None, show=True):
    """Loss: per-batch on the left, per-epoch train/val on the right."""
    edf, bdf, summary = load_run(run_dir)
    name = _pick_model(edf, model)
    e = _epoch_mean(edf, name)
    best = int(e["val_loss"].idxmin()) if "val_loss" in e else None
    with plt.rc_context(_STYLE):
        ncol = 2 if bdf is not None else 1
        fig, ax = plt.subplots(1, ncol, figsize=(6.6 * ncol, 4.6), squeeze=False)
        k = 0
        if bdf is not None:
            _batch_panel(ax[0][0], _batch_slice(bdf, name, member), "batch_loss",
                         "training loss on the batch", "Loss, batch by batch",
                         smooth=smooth, max_epoch=max_epoch)
            k = 1
        _epoch_panel(ax[0][k], e, [("train_loss", BLUE, "train"),
                                   ("val_loss", ORANGE, "validation")],
                     "cross-entropy loss", "Loss, epoch by epoch",
                     best_ep=best)
        fig.suptitle(_label(run_dir, summary, name), fontsize=11,
                     fontweight="bold", x=0.01, ha="left")
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        _finish(fig, out, show)
    return fig


def plot_auc(run_dir, model=None, member=0, out=None, smooth=None,
             max_epoch=None, show=True):
    """AUC: per-batch on the left, per-epoch train/val on the right."""
    edf, bdf, summary = load_run(run_dir)
    name = _pick_model(edf, model)
    e = _epoch_mean(edf, name)
    best = int(e["val_auc"].idxmax()) if "val_auc" in e else None
    with plt.rc_context(_STYLE):
        ncol = 2 if bdf is not None else 1
        fig, ax = plt.subplots(1, ncol, figsize=(6.6 * ncol, 4.6), squeeze=False)
        k = 0
        if bdf is not None:
            _batch_panel(ax[0][0], _batch_slice(bdf, name, member), "batch_auc",
                         "AUC on the batch", "AUC, batch by batch "
                         "(noisy: one batch is only 128 rows)",
                         smooth=smooth, max_epoch=max_epoch)
            k = 1
        _epoch_panel(ax[0][k], e, [("train_auc", BLUE, "train"),
                                   ("val_auc", ORANGE, "validation")],
                     "AUC", "AUC, epoch by epoch", best_ep=best)
        fig.suptitle(_label(run_dir, summary, name), fontsize=11,
                     fontweight="bold", x=0.01, ha="left")
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        _finish(fig, out, show)
    return fig


def plot_training_curves(run_dir, model=None, member=0, out=None, smooth=None,
                         max_epoch=None, show=True):
    """The 2x2 overview: loss and AUC, each at batch and epoch resolution."""
    edf, bdf, summary = load_run(run_dir)
    name = _pick_model(edf, model)
    e = _epoch_mean(edf, name)
    with plt.rc_context(_STYLE):
        ncol = 2 if bdf is not None else 1
        fig, ax = plt.subplots(2, ncol, figsize=(6.6 * ncol, 9), squeeze=False)
        k = 0
        if bdf is not None:
            b = _batch_slice(bdf, name, member)
            _batch_panel(ax[0][0], b, "batch_loss", "training loss on the batch",
                         "Loss, batch by batch", smooth=smooth, max_epoch=max_epoch)
            _batch_panel(ax[1][0], b, "batch_auc", "AUC on the batch",
                         "AUC, batch by batch (one batch = 128 rows)",
                         smooth=smooth, max_epoch=max_epoch)
            k = 1
        _epoch_panel(ax[0][k], e, [("train_loss", BLUE, "train"),
                                   ("val_loss", ORANGE, "validation")],
                     "cross-entropy loss", "Loss, epoch by epoch",
                     best_ep=int(e["val_loss"].idxmin()) if "val_loss" in e else None)
        _epoch_panel(ax[1][k], e, [("train_auc", BLUE, "train"),
                                   ("val_auc", ORANGE, "validation")],
                     "AUC", "AUC, epoch by epoch",
                     best_ep=int(e["val_auc"].idxmax()) if "val_auc" in e else None)
        fig.suptitle(_label(run_dir, summary, name), fontsize=11.5,
                     fontweight="bold", x=0.01, ha="left")
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        _finish(fig, out, show)
    return fig


def plot_capacity(run_dirs, out=None, show=True):
    """Compare several runs: test AUC against parameter count, per view."""
    rows = []
    for d in run_dirs:
        try:
            _, _, s = load_run(d)
        except FileNotFoundError:
            continue
        if not s:
            continue
        for r in s.get("results", []):
            rows.append(dict(arm=os.path.basename(os.path.normpath(d)),
                             views=r["views"], params=r.get("params"),
                             test_auc=r["test_auc"],
                             ceiling=s.get("tree_ceiling")))
    t = pd.DataFrame(rows).dropna(subset=["params"])
    if t.empty:
        raise ValueError("no finished runs with parameter counts found")
    with plt.rc_context(_STYLE):
        fig, a = plt.subplots(figsize=(8, 5))
        for views, color, marker, lab in [
                ("x+tree", BLUE, "o", "x + tree bits"),
                ("x", ORANGE, "s", "x only (no bits)")]:
            s_ = t[t["views"] == views].sort_values("params")
            if len(s_):
                a.plot(s_["params"], s_["test_auc"], marker + "-", color=color,
                       ms=8, lw=2, label=lab, mec="white", mew=1.2)
        ceil = t["ceiling"].dropna()
        if len(ceil):
            a.axhline(ceil.iloc[0], ls="--", color=MUTED, lw=1.2,
                      label=f"best tree {ceil.iloc[0]:.4f}")
        a.set_xscale("log")
        a.set_xlabel("trainable parameters (log scale)")
        a.set_ylabel("test AUC")
        a.set_title("Test AUC against model size", fontweight="bold", loc="left")
        a.legend(fontsize=9)
        fig.tight_layout()
        _finish(fig, out, show)
    return t


def plot_all(results_root="results/fusion", pattern="cap_*", out_dir=None,
             show=False):
    """Write a curve figure for every run under results_root."""
    made = []
    for d in sorted(glob.glob(os.path.join(results_root, pattern))):
        if not os.path.isdir(d):
            continue
        try:
            png = os.path.join(out_dir or d, "curves.png")
            plot_training_curves(d, out=png, show=show)
            made.append(png)
        except (FileNotFoundError, ValueError) as exc:
            print(f"  skipped {d}: {exc}")
    return made


def _finish(fig, out, show):
    if out:
        os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"[plot] -> {out}")
    if show:
        plt.show()
    else:
        plt.close(fig)


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", help="a run directory (results/fusion/<arm>)")
    ap.add_argument("--compare", help="glob of run directories to compare")
    ap.add_argument("--model", default=None,
                    help="which view config to plot (default: the x+tree one)")
    ap.add_argument("--member", type=int, default=0)
    ap.add_argument("--max-epoch", type=int, default=None,
                    help="zoom the batch panels to the first K epochs")
    ap.add_argument("--smooth", type=int, default=None,
                    help="rolling window in batches (default: half an epoch)")
    ap.add_argument("--what", default="both", choices=["both", "loss", "auc"])
    ap.add_argument("--out", default=None, help="output png")
    args = ap.parse_args()

    if args.compare:
        plot_capacity(sorted(glob.glob(args.compare)),
                      out=args.out or "capacity.png", show=False)
    elif args.run:
        kw = dict(model=args.model, member=args.member, smooth=args.smooth,
                  max_epoch=args.max_epoch, show=False)
        if args.what == "loss":
            plot_loss(args.run, out=args.out or os.path.join(args.run, "loss.png"), **kw)
        elif args.what == "auc":
            plot_auc(args.run, out=args.out or os.path.join(args.run, "auc.png"), **kw)
        else:
            plot_training_curves(
                args.run, out=args.out or os.path.join(args.run, "curves.png"), **kw)
    else:
        ap.error("pass --run or --compare")


if __name__ == "__main__":
    main()
