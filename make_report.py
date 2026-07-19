"""Turn a run's results_long.csv into paper-ready tables and figures.

Reads the long results (dataset x model x seed) produced by run_suite.py and
writes, into <out>/:

  main_results.(csv|md)   mean +/- std of the primary metric per dataset x model
  gap_analysis.csv        best-tree vs best-raw-NN vs best-TKCE, and % gap closed
  twostage_vs_joint.csv   the two TKCE regimes head to head
  kernel_ablation.csv     supervised GBT kernel vs unsupervised Mondrian kernel
  avg_rank.csv            average rank per model (via tkce.aggregate)
  cd_diagram.png          critical-difference (Nemenyi) diagram
  wilcoxon.csv            signed-rank test of each model vs --reference
  report.md               everything stitched together for the paper

Robust to partial results: only the models/datasets present are reported.
Metrics are unified into a higher-is-better `score` (AUC for classification,
R^2 for regression) so ranks work across mixed task types.

Usage:
  python make_report.py --results results/suite_v1/results_long.csv \
      --out results/suite_v1/report --reference catboost
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

from tkce.aggregate import full_report

FAMILIES = {
    "tree": ["xgboost", "lightgbm", "catboost", "random_forest"],
    "raw_nn": ["mlp_raw", "tabresnet_raw"],
    "strong_nn": ["ft_transformer", "num_embed_mlp"],
    "tree_feat": ["leafonehot_mlp", "pca_gbt_mlp", "pca_gbt_tabresnet"],
    "tkce_2stage": ["tkce2s_gbt_mlp", "tkce2s_gbt_tabresnet", "tkce2s_mondrian_mlp"],
    "tkce_joint": ["tkce_joint_gbt_mlp", "tkce_joint_gbt_tabresnet"],
}
TKCE_ALL = FAMILIES["tkce_2stage"] + FAMILIES["tkce_joint"]


def _fmt(m, s):
    return f"{m:.3f}±{s:.3f}" if np.isfinite(s) else f"{m:.3f}"


def _present(models, present):
    return [m for m in models if m in present]


def _best_in(row, models):
    vals = [row[m] for m in models if m in row and np.isfinite(row[m])]
    return max(vals) if vals else np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--reference", default="catboost")
    args = ap.parse_args()
    out = args.out or os.path.join(os.path.dirname(args.results), "report")
    os.makedirs(out, exist_ok=True)

    df = pd.read_csv(args.results)
    present = set(df["model"].unique())
    # Per (dataset, model): mean/std of the unified score.
    agg = df.groupby(["dataset", "model"])["score"].agg(["mean", "std"]).reset_index()
    metricmap = df.groupby("dataset")["primary"].first().to_dict()
    pivot_mean = agg.pivot(index="dataset", columns="model", values="mean")
    pivot_std = agg.pivot(index="dataset", columns="model", values="std")

    lines = ["# TKCE — Experimental Results", "",
             f"Datasets: {pivot_mean.shape[0]} | Models: {len(present)} | "
             f"Seeds aggregated. Score = AUC (classification) or R² (regression), "
             "higher is better.", ""]

    # ---- 1. Main results table ----
    ordered = (FAMILIES["tree"] + FAMILIES["raw_nn"] + FAMILIES["strong_nn"]
               + FAMILIES["tree_feat"] + TKCE_ALL)
    cols = _present(ordered, present)
    lines += ["## 1. Main results (mean ± std)", "",
              "Best per row in **bold**.", "",
              "| dataset | metric | " + " | ".join(cols) + " |",
              "|" + "---|" * (len(cols) + 2)]
    main_rows = []
    for dsname, row in pivot_mean.iterrows():
        best = _best_in(row, cols)
        cells = []
        rec = {"dataset": dsname, "metric": metricmap.get(dsname, "?")}
        for c in cols:
            m, s = row.get(c, np.nan), pivot_std.loc[dsname].get(c, np.nan)
            rec[c] = m
            if np.isfinite(m):
                txt = _fmt(m, s)
                cells.append(f"**{txt}**" if np.isclose(m, best) else txt)
            else:
                cells.append("—")
        main_rows.append(rec)
        lines.append(f"| {dsname} | {metricmap.get(dsname,'?')} | " + " | ".join(cells) + " |")
    pd.DataFrame(main_rows).to_csv(os.path.join(out, "main_results.csv"), index=False)

    # ---- 2. Gap analysis: does TKCE close the tree–NN gap? ----
    lines += ["", "## 2. Gap analysis — does TKCE close the tree→NN gap?", "",
              "`gap_closed = (best_TKCE − best_raw_NN) / (best_tree − best_raw_NN)` "
              "(only where trees actually beat the raw NN).", "",
              "| dataset | best_tree | best_raw_NN | best_strong_NN | best_TKCE | gap_closed |",
              "|---|---|---|---|---|---|"]
    gap_rows = []
    for dsname, row in pivot_mean.iterrows():
        bt = _best_in(row, _present(FAMILIES["tree"], present))
        bn = _best_in(row, _present(FAMILIES["raw_nn"], present))
        bs = _best_in(row, _present(FAMILIES["strong_nn"], present))
        bk = _best_in(row, _present(TKCE_ALL, present))
        gap = (bk - bn) / (bt - bn) if np.isfinite(bt) and np.isfinite(bn) and bt > bn else np.nan
        gap_rows.append({"dataset": dsname, "best_tree": bt, "best_raw_NN": bn,
                         "best_strong_NN": bs, "best_TKCE": bk, "gap_closed": gap})
        gp = f"{gap:.0%}" if np.isfinite(gap) else "—"
        lines.append(f"| {dsname} | {bt:.3f} | {bn:.3f} | "
                     f"{bs:.3f} | {bk:.3f} | {gp} |")
    gdf = pd.DataFrame(gap_rows)
    gdf.to_csv(os.path.join(out, "gap_analysis.csv"), index=False)
    med = gdf["gap_closed"].dropna()
    if len(med):
        lines += ["", f"Median gap closed by TKCE: **{med.median():.0%}** "
                  f"across {len(med)} datasets where trees led."]

    # ---- 3. Two-stage vs joint ----
    lines += ["", "## 3. Two-stage vs joint (the injection regime)", "",
              "| dataset | best_two_stage | best_joint | winner |", "|---|---|---|---|"]
    tj_rows = []
    for dsname, row in pivot_mean.iterrows():
        a = _best_in(row, _present(FAMILIES["tkce_2stage"], present))
        b = _best_in(row, _present(FAMILIES["tkce_joint"], present))
        win = "joint" if (np.isfinite(b) and (not np.isfinite(a) or b > a)) else "two-stage"
        tj_rows.append({"dataset": dsname, "two_stage": a, "joint": b, "winner": win})
        lines.append(f"| {dsname} | {a:.3f} | {b:.3f} | {win} |")
    pd.DataFrame(tj_rows).to_csv(os.path.join(out, "twostage_vs_joint.csv"), index=False)

    # ---- 4. Kernel ablation ----
    if {"tkce2s_gbt_mlp", "tkce2s_mondrian_mlp"} <= present:
        lines += ["", "## 4. Kernel ablation — supervised GBT vs unsupervised Mondrian", "",
                  "| dataset | GBT (target-led) | Mondrian (label-free) | Δ |",
                  "|---|---|---|---|"]
        ka_rows = []
        for dsname, row in pivot_mean.iterrows():
            g, mo = row.get("tkce2s_gbt_mlp", np.nan), row.get("tkce2s_mondrian_mlp", np.nan)
            ka_rows.append({"dataset": dsname, "gbt": g, "mondrian": mo, "delta": g - mo})
            lines.append(f"| {dsname} | {g:.3f} | {mo:.3f} | {g-mo:+.3f} |")
        pd.DataFrame(ka_rows).to_csv(os.path.join(out, "kernel_ablation.csv"), index=False)

    # ---- 5. Ranks / CD / Wilcoxon (delegated to aggregate) ----
    rep = full_report(df, "score", out, reference=args.reference)
    lines += ["", "## 5. Aggregate ranking", "",
              f"Critical difference (Nemenyi, α=0.05): **CD = {rep['cd']:.2f}** "
              f"over {pivot_mean.shape[0]} datasets. See `cd_diagram.png`.", "",
              "![CD diagram](cd_diagram.png)", "",
              "| model | avg_rank |", "|---|---|"]
    for m, r in rep["avg_rank"].items():
        lines.append(f"| {m} | {r:.2f} |")
    if rep["wilcoxon"] is not None:
        lines += ["", f"### Wilcoxon signed-rank vs `{args.reference}`", "",
                  "| model | mean_diff | wins | losses | p_value |", "|---|---|---|---|---|"]
        for _, r in rep["wilcoxon"].iterrows():
            lines.append(f"| {r['model']} | {r['mean_diff']:+.3f} | {int(r['wins'])} "
                         f"| {int(r['losses'])} | {r['p_value']:.3f} |")

    with open(os.path.join(out, "report.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[report] wrote {out}/report.md and CSVs "
          f"({pivot_mean.shape[0]} datasets, {len(present)} models)")


if __name__ == "__main__":
    main()
