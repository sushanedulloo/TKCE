"""How much signal is in the tree encoding? Ask a plain logistic regression.

A neural network can always be blamed for a result: the trunk might be doing
the work, the optimiser might not have converged, the capacity might be
"picking up anything". A logistic regression removes every one of those
objections at once. It has one weight per input feature, no hidden layer, no
nonlinearity, and a CONVEX objective — scikit-learn's solver returns the global
optimum, so there is no learning rate, no epoch count and no training
dynamics to argue about.

Whatever this model scores IS the linearly available signal in the input.

Three inputs are probed, under the usual OOB-honest protocol:

    x          the 10 raw features
    tree       ONLY the split bits — no raw features at all
    x + tree   both

with the tree baselines printed as the reference ceiling. The regularisation
strength C is chosen on the validation split, so nothing is tuned on test.

    python run_linear_probe.py --task 361055
    python run_linear_probe.py --task 361055 --rf-trees 100 --rf-depth 6
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from run_fusion import (fit_encoder_rf, node_depths_and_samples,
                        oob_honest_encoding, split_direction_encoding)
from tkce.baselines import fit_tree_baseline
from tkce.data import load_task

VIEWS = ["x", "tree", "x+tree"]


def build(view, X, T):
    if view == "x":
        return X
    if view == "tree":
        return T
    return np.concatenate([X, T], axis=1)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", type=int, default=361055)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-rows", type=int, default=16000)
    ap.add_argument("--rf-trees", type=int, default=100)
    ap.add_argument("--rf-depth", type=int, default=6)
    ap.add_argument("--rf-min-leaf", type=int, default=5)
    ap.add_argument("--encoding", default="oob", choices=["oob", "infold"])
    ap.add_argument("--views", default="x;tree;x+tree",
                    help="semicolon-separated inputs to probe, e.g. 'tree' to run "
                         "only the bits-alone arm (much faster)")
    ap.add_argument("--keep-layers", type=str, default=None,
                    help="keep only these tree depths, e.g. '0,1,2'. The columns for "
                         "every other split are DROPPED, so the reported feature "
                         "count is what the model actually sees")
    ap.add_argument("--drop-layers", type=str, default=None,
                    help="drop these tree depths, e.g. '5' or '4,5' for the last "
                         "few layers")
    ap.add_argument("--C", type=float, nargs="+",
                    default=[1e-4, 3e-4, 1e-3, 3e-3, 0.01, 0.03, 0.1, 0.3, 1.0],
                    help="inverse regularisation strengths (C = 1/lambda, so SMALL "
                         "C = strong regularisation); log-spaced in ~3x steps. The "
                         "best on VALIDATION is reported on test. The grid reaches "
                         "well below the useful range on purpose: with ~5,400 bits "
                         "and ~11,200 rows the optimum sits at the strongly "
                         "regularised end, and a grid that stops too early would "
                         "understate the signal")
    ap.add_argument("--max-iter", type=int, default=2000)
    ap.add_argument("--out", default="results/linear_probe")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    ds = load_task(args.task, seed=args.seed, max_rows=args.max_rows)
    print(f"\n=== LINEAR PROBE: {ds.name} (task {args.task}) ===")
    print(f"[data] train={len(ds.y_train)} val={len(ds.y_val)} test={len(ds.y_test)}"
          f" | {ds.X_train.shape[1]} raw features", flush=True)

    # ---- tree encoding, same protocol as run_fusion ----
    rf = fit_encoder_rf(ds.X_train, ds.y_train, args)
    T = {k: split_direction_encoding(rf, x).astype(np.float32)
         for k, x in [("train", ds.X_train), ("val", ds.X_val), ("test", ds.X_test)]}
    print(f"[view2] {T['train'].shape[1]} split bits "
          f"({args.rf_trees} trees, depth {args.rf_depth})", flush=True)
    if args.encoding == "oob":
        T["train"], mean_oob = oob_honest_encoding(rf, T["train"], len(ds.y_train))
        print(f"[view2] OOB-honest: each train row keeps bits from "
              f"{mean_oob:.1f}/{args.rf_trees} trees", flush=True)

    # ---- static bit selection (deletion is a plain column drop for a linear model) ----
    depths, samples = node_depths_and_samples(rf)
    kept_depths = sorted(set(depths.tolist()))
    if args.keep_layers and args.drop_layers:
        raise SystemExit("pass only one of --keep-layers / --drop-layers")
    sel = None
    if args.keep_layers:
        keep = [int(v) for v in args.keep_layers.split(",")]
        sel, what = np.isin(depths, keep), f"keep only depths {keep}"
    elif args.drop_layers:
        drop = [int(v) for v in args.drop_layers.split(",")]
        sel, what = ~np.isin(depths, drop), f"drop depths {drop}"
    if sel is not None:
        kept_depths = sorted(set(depths[sel].tolist()))
        print(f"[bits] {what}: keeping {int(sel.sum())}/{len(sel)} bits at depths "
              f"{kept_depths} (median {int(np.median(samples[sel]))} training rows "
              f"per kept split vs {int(np.median(samples[~sel]))} per dropped one)",
              flush=True)
        T = {k: v[:, sel] for k, v in T.items()}

    # ---- reference ceiling ----
    print("[ceiling] fitting tree baselines ...", flush=True)
    ceil = {}
    for name in ("xgboost", "lightgbm", "random_forest"):
        te, _, _ = fit_tree_baseline(name, ds, {"seed": args.seed})
        ceil[name] = te["auc"]
        print(f"  {name:14s} AUC={te['auc']:.4f}", flush=True)
    tree_ceiling = max(ceil.values())

    # ---- the probes ----
    results = []
    for view in [v.strip() for v in args.views.split(";") if v.strip()]:
        Xtr = build(view, ds.X_train, T["train"])
        Xva = build(view, ds.X_val, T["val"])
        Xte = build(view, ds.X_test, T["test"])
        print(f"\n[{view}] {Xtr.shape[1]} input features", flush=True)
        best = None
        for C in args.C:
            t0 = time.time()
            lr = LogisticRegression(C=C, max_iter=args.max_iter, n_jobs=-1)
            lr.fit(Xtr, ds.y_train)
            va = roc_auc_score(ds.y_val, lr.predict_proba(Xva)[:, 1])
            tr = roc_auc_score(ds.y_train, lr.predict_proba(Xtr)[:, 1])
            print(f"    C={C:<6g} train_auc={tr:.4f}  val_auc={va:.4f}"
                  f"   ({time.time() - t0:.1f}s)", flush=True)
            if best is None or va > best["val_auc"]:
                best = dict(C=C, val_auc=va, train_auc=tr,
                            test_auc=roc_auc_score(
                                ds.y_test, lr.predict_proba(Xte)[:, 1]))
        best.update(view=view, n_features=int(Xtr.shape[1]))
        # A winner at either end of the grid means the optimum may lie outside it,
        # so the reported score is a lower bound on what a linear model can do.
        lo, hi = min(args.C), max(args.C)
        edge = ("low" if best["C"] == lo and len(args.C) > 1 else
                "high" if best["C"] == hi and len(args.C) > 1 else None)
        best["C_at_grid_edge"] = edge
        results.append(best)
        print(f"  -> {view:8s} TEST auc={best['test_auc']:.4f} "
              f"(C={best['C']}, train {best['train_auc']:.4f}, "
              f"val {best['val_auc']:.4f})", flush=True)
        if edge == "low":
            print(f"     [!] best C is the SMALLEST tried ({lo:g}) — the optimum may "
                  f"be below the grid; rerun with smaller --C to be sure", flush=True)
        elif edge == "high":
            print(f"     [!] best C is the LARGEST tried ({hi:g}) — the optimum may "
                  f"be above the grid; rerun with larger --C to be sure", flush=True)

    # ---- verdict ----
    by = {r["view"]: r for r in results}
    print("\n" + "=" * 64)
    print(f"LINEAR PROBE — {ds.name}  (logistic regression, converged, no SGD)")
    print("=" * 64)
    print(f"  {'input':10s} {'features':>9s} {'train':>8s} {'val':>8s} {'test':>8s}")
    for r in results:
        print(f"  {r['view']:10s} {r['n_features']:9d} {r['train_auc']:8.4f} "
              f"{r['val_auc']:8.4f} {r['test_auc']:8.4f}")
    print(f"  {'best tree':10s} {'':9s} {'':8s} {'':8s} {tree_ceiling:8.4f}")
    edged = [r["view"] for r in results if r.get("C_at_grid_edge")]
    if edged:
        print(f"\n  [!] C landed on the edge of the grid for: {', '.join(edged)}"
              f"  (grid was {min(args.C):g} .. {max(args.C):g})")
    print()
    if "tree" in by:
        print(f"  tree bits alone, with NO raw features and NO hidden layer: "
              f"{by['tree']['test_auc']:.4f}")
        print(f"  fraction of the tree ceiling reached by the bits alone:    "
              f"{by['tree']['test_auc'] / tree_ceiling:.1%}")
    if "x" in by:
        print(f"  raw features alone, same linear model:                     "
              f"{by['x']['test_auc']:.4f}")
    if "tree" in by and "x" in by:
        print(f"  the bits are worth "
              f"{by['tree']['test_auc'] - by['x']['test_auc']:+.4f} AUC to a "
              f"linear model")

    summary = dict(dataset=ds.name, task=args.task, seed=args.seed,
                   encoding=args.encoding, rf_trees=args.rf_trees,
                   rf_depth=args.rf_depth, tree_encoding_width=int(T["val"].shape[1]),
                   keep_layers=args.keep_layers, drop_layers=args.drop_layers,
                   kept_depths=kept_depths, C_grid=list(args.C),
                   ceiling=ceil, tree_ceiling=tree_ceiling, results=results)
    path = os.path.join(args.out, f"linear_probe_{ds.name}.json")
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[probe] summary -> {path}")


if __name__ == "__main__":
    main()
