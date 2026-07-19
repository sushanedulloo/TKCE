"""Lambda sweep: how much should the joint regime weight the contrastive term?

The joint loss is task_loss + lambda * contrastive_loss. We sweep lambda for
tkce_joint_gbt_mlp; lambda=0 is the joint architecture with NO tree bias (a
plain NN), and larger lambda pulls the representation toward the tree kernel.
A peak above lambda=0 shows the contrastive term helps; the location is the
sweet spot. Raw-MLP and best-tree references are drawn as horizontal lines.

Usage:
  python run_lambda_sweep.py --tasks 361070 --seeds 0,1,2 --device auto \
      --out paper/figures
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
from tkce.metrics import primary_metric
from tkce.train import resolve_device


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default="361070")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--max-rows", type=int, default=8000)
    ap.add_argument("--lambdas", default="0,0.1,0.3,1,3,10")
    ap.add_argument("--out", default="paper/figures")
    ap.add_argument("--csv", default="results/analysis")
    args = ap.parse_args()

    tasks = [int(t) for t in args.tasks.split(",")]
    seeds = [int(s) for s in args.seeds.split(",")]
    lambdas = [float(x) for x in args.lambdas.split(",")]
    device = resolve_device(args.device)
    os.makedirs(args.out, exist_ok=True); os.makedirs(args.csv, exist_ok=True)

    rows, refs = [], []
    meta = {}
    for task in tasks:
        for seed in seeds:
            base = load_task(task, seed=seed, max_rows=args.max_rows)
            meta[task] = (base.name, primary_metric(base.task_type))
            for lam in lambdas:
                sc = evaluate("tkce_joint_gbt_mlp", base, device, seed,
                              cfg={"lambda_contrast": lam})
                rows.append(dict(task=task, dataset=base.name, lam=lam, seed=seed, score=sc))
            # references
            refs.append(dict(task=task, model="mlp_raw", seed=seed,
                             score=evaluate("mlp_raw", base, device, seed)))
            refs.append(dict(task=task, model="xgboost", seed=seed,
                             score=evaluate("xgboost", base, device, seed)))
            print(f"[lambda] task {task} ({base.name}) seed {seed} done")

    df = pd.DataFrame(rows); rdf = pd.DataFrame(refs)
    df.to_csv(os.path.join(args.csv, "lambda_sweep.csv"), index=False)

    fig, axes = plt.subplots(1, len(tasks), figsize=(6 * len(tasks), 4.2), squeeze=False)
    for ax, task in zip(axes[0], tasks):
        name, pm = meta[task]
        sub = df[df.task == task]
        mean = [sub[sub.lam == l]["score"].mean() for l in lambdas]
        std = [sub[sub.lam == l]["score"].std(ddof=0) for l in lambdas]
        xs = [max(l, 0.03) for l in lambdas]  # so lambda=0 shows on a log axis
        ax.plot(xs, mean, marker="o", label="TKCE joint")
        ax.fill_between(xs, np.array(mean) - np.nan_to_num(std),
                        np.array(mean) + np.nan_to_num(std), alpha=0.15)
        for model, style in [("mlp_raw", "--"), ("xgboost", ":")]:
            v = rdf[(rdf.task == task) & (rdf.model == model)]["score"].mean()
            ax.axhline(v, ls=style, color="grey",
                       label=f"{model} ({v:.3f})")
        ax.set_xscale("log"); ax.set_xlabel("λ (contrastive weight)")
        ax.set_ylabel(f"test {pm}"); ax.set_title(name)
        ax.grid(True, alpha=0.3); ax.legend(fontsize=8)
    fig.suptitle("Joint regime: effect of the contrastive weight λ",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    out_png = os.path.join(args.out, "lambda_sweep.png")
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"[lambda] figure -> {out_png} | data -> {args.csv}/lambda_sweep.csv")


if __name__ == "__main__":
    main()
