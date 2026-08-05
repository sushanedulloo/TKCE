"""Is the *tree* ceiling itself overfitting? Report train vs val vs test AUC.

If the trees also have train AUC ~1.0 but test ~0.70, then 0.70 is the genuine
generalization ceiling of this dataset (everyone hits the same wall), not a
neural-net-specific problem.

    python check_tree_overfit.py --task 361070
"""

from __future__ import annotations

import argparse

from tkce.baselines import fit_tree_baseline
from tkce.data import load_task
from tkce.metrics import clf_metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", type=int, default=361070)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-rows", type=int, default=16000)
    args = ap.parse_args()

    ds = load_task(args.task, seed=args.seed, max_rows=args.max_rows)
    print(f"\n=== TREE overfitting check: {ds.name} (task {args.task}) ===")
    print(f"    train={len(ds.y_train)}  val={len(ds.y_val)}  test={len(ds.y_test)} rows\n")
    print(f"    {'model':16s} {'train':>7} {'val':>7} {'test':>7}   {'gap(tr-te)':>10}")
    for name in ("xgboost", "lightgbm", "random_forest"):
        test, val, model = fit_tree_baseline(name, ds, {"seed": args.seed})
        tr = clf_metrics(ds.y_train, model.predict_proba(ds.X_train))
        gap = tr["auc"] - test["auc"]
        print(f"    {name:16s} {tr['auc']:7.4f} {val['auc']:7.4f} {test['auc']:7.4f}   "
              f"{gap:+10.4f}")
    print("\n  Reading: big train-test gap => the tree ALSO overfits; test AUC is the")
    print("  real signal ceiling. Small gap => the tree generalizes well.\n")


if __name__ == "__main__":
    main()
