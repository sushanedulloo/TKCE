"""Mechanism probes: does TKCE inherit the THREE tree advantages (Grinsztajn 2022)?

Three controlled experiments, one per panel of the figure:
  (a) Uninformative features — append k noise columns; robust models degrade less.
  (b) Rotation — rotate the feature space; axis-aligned models (trees, and we
      hope TKCE) degrade while a rotation-invariant MLP stays flat.
  (c) Irregular targets — a checkerboard target of increasing frequency; trees
      fit it, smooth NNs struggle.

Panels (a,b) perturb a real numerical dataset; (c) uses a synthetic checkerboard.
All classification (metric = AUC) so the y-axis is comparable across panels.

Usage:
  python run_mechanism.py --task 361065 --seeds 0,1,2 --device auto \
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

from tkce.analysis import (MODELS_DEFAULT, add_noise_features, evaluate,
                           line_plot, make_dataset_from_arrays,
                           make_synthetic_irregular, rotate_features)
from tkce.data import load_task
from tkce.train import resolve_device


def _agg(df, models, xcol):
    xs = sorted(df[xcol].unique())
    series = {}
    for m in models:
        sub = df[df.model == m]
        mean = [sub[sub[xcol] == x]["score"].mean() for x in xs]
        std = [sub[sub[xcol] == x]["score"].std(ddof=0) for x in xs]
        series[m] = (np.array(mean), np.nan_to_num(np.array(std)))
    return xs, series


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", type=int, default=361065)  # MagicTelescope (numerical clf)
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--max-rows", type=int, default=6000)
    ap.add_argument("--models", default=",".join(MODELS_DEFAULT))
    ap.add_argument("--noise-ks", default="0,5,10,25,50,100")
    ap.add_argument("--angles", default="0,15,30,45,60,90")
    ap.add_argument("--freqs", default="1,2,3,4,6,8")
    ap.add_argument("--n-synth", type=int, default=4000)
    ap.add_argument("--p-synth", type=int, default=4)
    ap.add_argument("--out", default="paper/figures")
    ap.add_argument("--csv", default="results/analysis")
    args = ap.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    models = args.models.split(",")
    noise_ks = [int(x) for x in args.noise_ks.split(",")]
    angles = [float(x) for x in args.angles.split(",")]
    freqs = [int(x) for x in args.freqs.split(",")]
    device = resolve_device(args.device)
    os.makedirs(args.out, exist_ok=True); os.makedirs(args.csv, exist_ok=True)

    rows = []
    for seed in seeds:
        base = load_task(args.task, seed=seed, max_rows=args.max_rows)
        print(f"[mechanism] seed {seed}: base={base.name} ({base.task_type})")
        # (a) uninformative features
        for k in noise_ks:
            ds = add_noise_features(base, k, seed)
            for m in models:
                rows.append(dict(panel="a_noise", x=k, model=m, seed=seed,
                                 score=evaluate(m, ds, device, seed)))
        # (b) rotation
        for ang in angles:
            ds = rotate_features(base, ang, seed)
            for m in models:
                rows.append(dict(panel="b_rotate", x=ang, model=m, seed=seed,
                                 score=evaluate(m, ds, device, seed)))
        # (c) irregular targets (synthetic)
        for fq in freqs:
            X, y = make_synthetic_irregular(args.n_synth, args.p_synth, fq, seed)
            ds = make_dataset_from_arrays(X, y, "classification", seed,
                                          name=f"checker_f{fq}")
            for m in models:
                rows.append(dict(panel="c_irregular", x=fq, model=m, seed=seed,
                                 score=evaluate(m, ds, device, seed)))
        print(f"[mechanism] seed {seed} done")

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(args.csv, "mechanism.csv"), index=False)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for ax, (panel, xl, ttl) in zip(axes, [
            ("a_noise", "# uninformative features added", "(a) Robustness to noise features"),
            ("b_rotate", "feature rotation angle (deg)", "(b) Sensitivity to rotation"),
            ("c_irregular", "target frequency (irregularity)", "(c) Irregular targets")]):
        xs, series = _agg(df[df.panel == panel], models, "x")
        line_plot(ax, xs, series, xl, "test AUC", ttl)
    fig.suptitle("Mechanism probes: does TKCE inherit the tree advantages?",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    out_png = os.path.join(args.out, "mechanism.png")
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"[mechanism] figure -> {out_png} | data -> {args.csv}/mechanism.csv")


if __name__ == "__main__":
    main()
