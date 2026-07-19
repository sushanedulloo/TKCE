"""Contrastive-loss ablation: which loss learns the best TKCE embedding?

Runs two-stage TKCE (GBT kernel) with each contrastive loss and reports the
downstream metric per loss, per dataset. Raw-MLP and best-tree references give
context (a loss is only useful if its embedding beats raw features).

Losses: infonce, kernel_regression, contrastive, triplet, supcon, aninfonce,
clip_infonce.

Usage:
  python run_loss_ablation.py --tasks 361070,361072 --seeds 0,1,2 --device auto
"""

from __future__ import annotations

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tkce.analysis import evaluate
from tkce.data import load_task
from tkce.losses import ALL_LOSSES
from tkce.metrics import primary_metric
from tkce.train import resolve_device


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default="361070,361072")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--max-rows", type=int, default=8000)
    ap.add_argument("--model", default="tkce2s_gbt_mlp",
                    help="TKCE two-stage variant whose loss is swept")
    ap.add_argument("--losses", default=",".join(ALL_LOSSES))
    ap.add_argument("--out", default="paper/figures")
    ap.add_argument("--csv", default="results/analysis")
    args = ap.parse_args()

    tasks = [int(t) for t in args.tasks.split(",")]
    seeds = [int(s) for s in args.seeds.split(",")]
    losses = args.losses.split(",")
    device = resolve_device(args.device)
    os.makedirs(args.out, exist_ok=True); os.makedirs(args.csv, exist_ok=True)

    rows, refs, meta = [], [], {}
    for task in tasks:
        for seed in seeds:
            ds = load_task(task, seed=seed, max_rows=args.max_rows)
            meta[task] = (ds.name, primary_metric(ds.task_type))
            for L in losses:
                try:
                    sc = evaluate(args.model, ds, device, seed,
                                  cfg={"contrastive_loss": L})
                except Exception as e:  # noqa: BLE001
                    print(f"  {ds.name} seed{seed} {L} FAILED: {e}"); sc = np.nan
                rows.append(dict(task=task, dataset=ds.name, loss=L, seed=seed, score=sc))
            for m in ("mlp_raw", "xgboost"):
                refs.append(dict(task=task, model=m, seed=seed,
                                 score=evaluate(m, ds, device, seed)))
            print(f"[loss-abl] task {task} ({ds.name}) seed {seed} done")

    df = pd.DataFrame(rows); rdf = pd.DataFrame(refs)
    df.to_csv(os.path.join(args.csv, "loss_ablation.csv"), index=False)

    fig, axes = plt.subplots(1, len(tasks), figsize=(1.6 * len(losses) * len(tasks), 4.4),
                             squeeze=False)
    for ax, task in zip(axes[0], tasks):
        name, pm = meta[task]
        sub = df[df.task == task]
        mean = [sub[sub.loss == L]["score"].mean() for L in losses]
        std = [sub[sub.loss == L]["score"].std(ddof=0) for L in losses]
        ax.bar(range(len(losses)), mean, yerr=np.nan_to_num(std), capsize=3,
               color="steelblue", edgecolor="white")
        for m, style in [("mlp_raw", "--"), ("xgboost", ":")]:
            v = rdf[(rdf.task == task) & (rdf.model == m)]["score"].mean()
            ax.axhline(v, ls=style, color="grey", lw=1.2, label=f"{m} ({v:.3f})")
        ax.set_xticks(range(len(losses)))
        ax.set_xticklabels(losses, rotation=35, ha="right", fontsize=8)
        ax.set_ylabel(f"test {pm}"); ax.set_title(name)
        ax.legend(fontsize=8); ax.grid(True, axis="y", alpha=0.3)
    fig.suptitle("Contrastive-loss ablation (two-stage TKCE embedding)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    out_png = os.path.join(args.out, "loss_ablation.png")
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"[loss-abl] figure -> {out_png} | data -> {args.csv}/loss_ablation.csv")


if __name__ == "__main__":
    main()
