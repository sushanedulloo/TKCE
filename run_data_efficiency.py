"""Data-efficiency curves: does TKCE help the NN most when data is scarce?

Trees dominate in the low-data regime. We train every model on increasing
fractions of the training set and plot the primary metric vs. fraction. If TKCE
sits above the raw NN especially at small fractions, it inherits the tree's
low-data strength.

Usage:
  python run_data_efficiency.py --tasks 361070,361072 --seeds 0,1,2 \
      --device auto --out paper/figures
"""

from __future__ import annotations

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tkce.analysis import (MODELS_DEFAULT, evaluate, line_plot, subsample_train)
from tkce.data import load_task
from tkce.metrics import primary_metric
from tkce.train import resolve_device


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default="361070,361072")  # eye_movements(clf), cpu_act(reg)
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--max-rows", type=int, default=8000)
    ap.add_argument("--models", default=",".join(MODELS_DEFAULT))
    ap.add_argument("--fracs", default="0.05,0.1,0.25,0.5,1.0")
    ap.add_argument("--out", default="paper/figures")
    ap.add_argument("--csv", default="results/analysis")
    args = ap.parse_args()

    tasks = [int(t) for t in args.tasks.split(",")]
    seeds = [int(s) for s in args.seeds.split(",")]
    models = args.models.split(",")
    fracs = [float(f) for f in args.fracs.split(",")]
    device = resolve_device(args.device)
    os.makedirs(args.out, exist_ok=True); os.makedirs(args.csv, exist_ok=True)

    rows = []
    task_meta = {}
    for task in tasks:
        for seed in seeds:
            base = load_task(task, seed=seed, max_rows=args.max_rows)
            task_meta[task] = (base.name, primary_metric(base.task_type))
            for fr in fracs:
                ds = subsample_train(base, fr, seed)
                for m in models:
                    rows.append(dict(task=task, dataset=base.name, frac=fr, model=m,
                                     seed=seed, score=evaluate(m, ds, device, seed)))
            print(f"[data-eff] task {task} ({base.name}) seed {seed} done")

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(args.csv, "data_efficiency.csv"), index=False)

    fig, axes = plt.subplots(1, len(tasks), figsize=(6 * len(tasks), 4.2), squeeze=False)
    for ax, task in zip(axes[0], tasks):
        sub = df[df.task == task]
        name, pm = task_meta[task]
        series = {}
        for m in models:
            s = sub[sub.model == m]
            mean = [s[s.frac == f]["score"].mean() for f in fracs]
            std = [s[s.frac == f]["score"].std(ddof=0) for f in fracs]
            series[m] = (np.array(mean), np.nan_to_num(np.array(std)))
        line_plot(ax, fracs, series, "training-set fraction", f"test {pm}",
                  f"{name}", logx=True)
    fig.suptitle("Data efficiency: TKCE vs. trees vs. raw NN",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    out_png = os.path.join(args.out, "data_efficiency.png")
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"[data-eff] figure -> {out_png} | data -> {args.csv}/data_efficiency.csv")


if __name__ == "__main__":
    main()
