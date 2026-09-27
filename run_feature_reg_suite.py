"""Does the feature-regularisation observation hold on other datasets?

Runs the SAME fixed set of arms, with every experimental factor held identical,
across datasets of three task types, then aggregates one table and one figure
that answer the question directly.

    binary          credit (anchor), house_16H, MagicTelescope
    multi-class     pendigits (10 classes), covertype (7 classes, capped rows)
    regression      cpu_act, wine_quality

Held fixed everywhere: forest of 100 trees at depth 6 and min-leaf 5, the
out-of-bag honest protocol, a convex linear head at C = 0.01 (Ridge alpha =
1/(2C) for regression), three stacked noisy copies for the noise arms, at most
16,000 rows, seed 0. The arms are the ones that carried the credit result:

    baseline                 all bits, no perturbation (+ the raw-x floor)
    uniform p = 0.2          noise at the level that was best on credit
    ramp P = 0.2             depth-increasing flips
    uniform p = 0.16         the matched-budget control for that ramp
    drop depth 5             delete the deepest layer
    drop depths 4,5          delete the two deepest layers
    drop 4,5 + ramp 0.2      the combined design
    keep depths 0-2          the recommended configuration from credit

Every metric is tracked on train (clean bits), validation and test. The suite
is resumable: a (dataset, arm) whose JSON already exists is skipped.

    python run_feature_reg_suite.py                         # everything
    python run_feature_reg_suite.py --datasets credit,cpu_act
    python run_feature_reg_suite.py --quick                 # smoke settings
    python run_feature_reg_suite.py --aggregate-only        # tables + figure
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys
import time

import numpy as np
import pandas as pd

DATASETS = [
    # (name, OpenML task id, kind) -- kind is only a label for the tables
    ("credit",         361055, "binary"),
    ("house_16H",      361063, "binary"),
    ("MagicTelescope", 361065, "binary"),
    ("pendigits",          32, "multiclass"),
    ("covertype",        7593, "multiclass"),
    ("cpu_act",        361072, "regression"),
    ("wine_quality",   361076, "regression"),
]

ARMS = [
    # (label, extra flags). Labels sort into the order the tables should show.
    ("00_baseline",         ""),
    ("10_unif_0.2",         "--flip-uniform 0.2"),
    ("20_ramp_0.2",         "--flip-ramp 0.2"),
    ("21_ctrl_0.16",        "--flip-uniform 0.16"),
    ("30_drop_L5",          "--drop-layers 5"),
    ("31_drop_L45",         "--drop-layers 4,5"),
    ("33_drop_L45_ramp02",  "--drop-layers 4,5 --flip-ramp 0.2"),
    ("40_keep_L012",        "--keep-layers 0,1,2"),
]

SHARED = ("--C 0.01 --encoding oob --noise-copies 3 --rf-trees 100 --rf-depth 6 "
          "--rf-min-leaf 5 --seed 0 --max-rows 16000")
QUICK = "--max-rows 2500 --rf-trees 25 --max-iter 300 --noise-copies 2"

PY = sys.executable


# --------------------------------------------------------------------------- #
# running
# --------------------------------------------------------------------------- #
def run_all(root, datasets, arms, quick):
    todo = [(d, a) for d in datasets for a in arms]
    print(f"[suite] {len(datasets)} datasets x {len(arms)} arms = {len(todo)} runs "
          f"-> {root}", flush=True)
    t_all = time.time()
    for i, ((name, task, kind), (label, flags)) in enumerate(todo, 1):
        out = os.path.join(root, name, label)
        if glob.glob(os.path.join(out, "featreg_*.json")):
            print(f"[suite] {i:2d}/{len(todo)} {name:15s} {label:20s} done, skipping",
                  flush=True)
            continue
        # The baseline also runs the raw-x floor; every other arm is bits only.
        views = "x;tree" if label == "00_baseline" else "tree"
        cmd = [PY, "-u", "run_feature_reg.py", "--task", str(task), "--views", views,
               "--label", label, "--out", out] + SHARED.split()
        if quick:
            cmd += QUICK.split()
        cmd += flags.split()
        print(f"\n[suite] {i:2d}/{len(todo)} {name} ({kind}) :: {label}  "
              f"[{time.strftime('%H:%M:%S')}]", flush=True)
        t0 = time.time()
        r = subprocess.run(cmd)
        status = "ok" if r.returncode == 0 else f"FAILED (exit {r.returncode})"
        print(f"[suite] {name} :: {label} -> {status} in {time.time() - t0:.0f}s",
              flush=True)
    print(f"\n[suite] all runs finished in {(time.time() - t_all) / 60:.1f} min",
          flush=True)


# --------------------------------------------------------------------------- #
# aggregating
# --------------------------------------------------------------------------- #
def load_rows(root):
    rows = []
    for js in sorted(glob.glob(os.path.join(root, "*", "*", "featreg_*.json"))):
        s = json.load(open(js))
        prim = s["primary"]
        for r in s["results"]:
            row = dict(dataset=s["dataset"], task=s["task"], task_type=s["task_type"],
                       n_classes=s["n_classes"], primary=prim, arm=s["label"],
                       view=r["view"], bits=r["n_features"],
                       kept_depths=str(s["kept_depths"]),
                       mean_p=round(s["flip_mean_p"], 4),
                       train=r["train"][prim], val=r["val"][prim], test=r["test"][prim],
                       gap=r["gap"], tree_ceiling=s["tree_ceiling"],
                       best_tree=s["best_tree"], seconds=r["seconds"])
            # the secondary metrics, so "all metrics across train, val and test"
            # is literally true in the CSV
            for split in ("train", "val", "test"):
                for k, v in r[split].items():
                    if k != prim:
                        row[f"{split}_{k}"] = v
            rows.append(row)
    return pd.DataFrame(rows)


def verdict_table(t):
    """One row per dataset: does the credit observation hold here?"""
    out = []
    for ds, g in t[t.view == "tree"].groupby("dataset", sort=False):
        g = g.set_index("arm")
        if "00_baseline" not in g.index:
            continue
        b = g.loc["00_baseline"]
        floor = t[(t.dataset == ds) & (t.view == "x")]
        row = dict(dataset=ds, type=b["task_type"] + (f" ({int(b['n_classes'])} cls)"
                                                       if b["n_classes"] > 2 else ""),
                   metric=b["primary"], tree=round(b["tree_ceiling"], 4),
                   raw_x=round(float(floor.test.iloc[0]), 4) if len(floor) else None,
                   base_train=round(b["train"], 4), base_test=round(b["test"], 4),
                   base_gap=round(b["gap"], 4))
        if "40_keep_L012" in g.index:
            k = g.loc["40_keep_L012"]
            row.update(keep012_bits=int(k["bits"]), keep012_train=round(k["train"], 4),
                       keep012_test=round(k["test"], 4), keep012_gap=round(k["gap"], 4),
                       d_test=round(k["test"] - b["test"], 4),
                       # a ratio is only meaningful when the baseline actually overfits
                       gap_ratio=(round(k["gap"] / b["gap"], 2)
                                  if b["gap"] > 0.005 else None))
            # The observation: test within 0.01 of the baseline AND the gap shrinks.
            row["holds"] = ("yes" if (k["test"] >= b["test"] - 0.01 and k["gap"] < b["gap"])
                            else "no")
        if "20_ramp_0.2" in g.index and "21_ctrl_0.16" in g.index:
            row["ramp_minus_ctrl"] = round(g.loc["20_ramp_0.2", "test"]
                                           - g.loc["21_ctrl_0.16", "test"], 4)
        # best arm by test, and the arm with the smallest gap that is within 0.01
        row["best_arm"] = g["test"].idxmax()
        ok = g[g["test"] >= b["test"] - 0.01]
        row["lowest_gap_arm_ok"] = ok["gap"].idxmin() if len(ok) else None
        out.append(row)
    return pd.DataFrame(out)


def figure(t, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    BLUE, ORANGE, AQUA, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7"
    AMBER, MUTED = "#b5730f", "#52514e"
    plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": .22,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "figure.facecolor": "white", "legend.frameon": False})
    tt = t[t.view == "tree"].copy()
    arms = [a for a, _ in ARMS if a in set(tt.arm)]
    dsets = list(dict.fromkeys(tt.dataset))
    palette = [BLUE, ORANGE, AQUA, VIOLET, AMBER, "#e34948", "#008300", MUTED]
    fig, ax = plt.subplots(1, 2, figsize=(13, 5.2))
    for a_, col, ttl, ylab, hline in [
            (ax[0], "d_test", "Change in TEST metric vs. baseline\n(within the grey band = no cost)",
             "test minus baseline (AUC or R$^2$)", 0.0),
            (ax[1], "gap_ratio", "Overfitting gap relative to baseline\n(below 1 = less overfitting)",
             "gap / baseline gap", 1.0)]:
        for i, ds in enumerate(dsets):
            g = tt[tt.dataset == ds].set_index("arm")
            if "00_baseline" not in g.index:
                continue
            b = g.loc["00_baseline"]
            ys = []
            for a in arms:
                if a not in g.index:
                    ys.append(np.nan); continue
                ys.append((g.loc[a, "test"] - b["test"]) if col == "d_test"
                          else (g.loc[a, "gap"] / b["gap"] if b["gap"] else np.nan))
            a_.plot(range(len(arms)), ys, "o-", lw=1.6, ms=6, color=palette[i % len(palette)],
                    label=f"{ds} ({g.iloc[0]['task_type'][:3]})", alpha=.9)
        if col == "d_test":
            a_.axhspan(-0.01, 0.01, color="#c8c8c4", alpha=.35, lw=0)
        a_.axhline(hline, color=MUTED, lw=1.1, ls="--")
        a_.set_xticks(range(len(arms)))
        a_.set_xticklabels([a.split("_", 1)[1] for a in arms], rotation=30, ha="right")
        a_.set_ylabel(ylab)
        a_.set_title(ttl, fontweight="bold", loc="left", fontsize=10)
    ax[0].legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f"[suite] figure -> {path}")


def aggregate(root):
    t = load_rows(root)
    if t.empty:
        print("[suite] nothing to aggregate yet")
        return
    t.to_csv(os.path.join(root, "suite_long.csv"), index=False)
    v = verdict_table(t)
    v.to_csv(os.path.join(root, "suite_verdict.csv"), index=False)
    pd.set_option("display.width", 250)
    print("\n" + "=" * 100)
    print("PER-DATASET VERDICT  (train / test on the primary metric; gap = train - val on CLEAN bits)")
    print("=" * 100)
    print(v.to_string(index=False))
    print("\n  holds = keep-depths-0-2 test within 0.01 of the baseline AND a smaller gap")
    print("  gap_ratio is blank when the baseline gap is under 0.005 (nothing to reduce); "
          "a gap <= 0 means no measurable overfitting")
    print("  ramp_minus_ctrl = ramp 0.2 test minus matched uniform 0.16 test "
          "(<= 0 means the depth shape did not help)")
    print("\nFULL LONG TABLE (every arm, every split, every metric) -> suite_long.csv")
    figure(t, os.path.join(root, "suite_overview.png"))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="results/featreg_suite")
    ap.add_argument("--datasets", default=None,
                    help="comma-separated subset of dataset names")
    ap.add_argument("--arms", default=None, help="comma-separated subset of arm labels")
    ap.add_argument("--quick", action="store_true", help="smoke-test settings")
    ap.add_argument("--aggregate-only", action="store_true")
    args = ap.parse_args()

    datasets = DATASETS
    if args.datasets:
        want = set(args.datasets.split(","))
        datasets = [d for d in DATASETS if d[0] in want]
    arms = ARMS
    if args.arms:
        want = set(args.arms.split(","))
        arms = [a for a in ARMS if a[0] in want]
    os.makedirs(args.root, exist_ok=True)
    if not args.aggregate_only:
        run_all(args.root, datasets, arms, args.quick)
    aggregate(args.root)


if __name__ == "__main__":
    main()
