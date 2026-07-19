"""Full experiment driver: tune (equal budget) -> multi-seed eval -> aggregate.

Protocol (Grinsztajn-style):
  for each dataset:
    load the seed-0 split
    for each model: tune HPs on (train, val) with `--trials` Optuna trials
                    then evaluate the winning cfg on every seed's test split
  aggregate: per-dataset ranks, average rank, Wilcoxon vs reference, CD diagram.

Mixed classification (AUC) and regression (R^2) datasets are unified into a
single higher-is-better `score` column so ranks are computed within each dataset.

Usage:
  python run_suite.py --tasks 361070,361062 --seeds 0,1,2 --trials 100 \
      --max-rows 8000 --device mps --out results/suite_run1
"""

from __future__ import annotations

import argparse
import json
import os
import time

import pandas as pd

from tkce.aggregate import full_report
from tkce.data import load_task
from tkce.metrics import primary_metric
from tkce.train import resolve_device
from tkce.tuning import DEFAULT_MODELS, run_model, tune_model

# Curated subset with genuine tree>NN gaps (numeric clf + reg). Override with --tasks.
DEFAULT_TASKS = [361070, 361062, 361063]  # eye_movements, pol, house_16H (clf)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default=",".join(map(str, DEFAULT_TASKS)))
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--trials", type=int, default=100)
    ap.add_argument("--max-rows", type=int, default=8000)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--models", default="all")
    ap.add_argument("--reference", default="catboost")
    ap.add_argument("--out", default="results/suite_run")
    args = ap.parse_args()

    tasks = [int(x) for x in args.tasks.split(",")]
    seeds = [int(x) for x in args.seeds.split(",")]
    models = DEFAULT_MODELS if args.models == "all" else args.models.split(",")
    device = resolve_device(args.device)
    os.makedirs(args.out, exist_ok=True)
    long_path = os.path.join(args.out, "results_long.csv")
    print(f"[suite] tasks={tasks} seeds={seeds} trials={args.trials} "
          f"models={len(models)} device={device}")

    # Resume: reload any prior rows and skip (task, model) cells already complete.
    rows = []
    done = set()  # (task_id, model) with all requested seeds present
    if os.path.exists(long_path):
        prev = pd.read_csv(long_path)
        rows = prev.to_dict("records")
        for (tid, mdl), grp in prev.groupby(["task_id", "model"]):
            if set(seeds).issubset(set(grp["seed"].tolist())):
                done.add((tid, mdl))
        if done:
            print(f"[suite] resuming: {len(done)} (task,model) cells already done")

    for task in tasks:
        t_task = time.time()
        ds0 = load_task(task, seed=seeds[0], max_rows=args.max_rows)
        pm = primary_metric(ds0.task_type)
        cache = {seeds[0]: ds0}
        print(f"\n[suite] === {ds0.name} (task {task}, {ds0.task_type}, "
              f"metric={pm}, n={ds0.meta['n_rows']}) ===")

        for model in models:
            if (task, model) in done:
                print(f"  {model:26s} (skipped, already done)")
                continue
            t0 = time.time()
            try:
                best_cfg, best_val, _ = tune_model(model, ds0, args.trials, device,
                                                   seed=seeds[0])
            except Exception as e:  # noqa: BLE001
                print(f"  [tune] {model:26s} FAILED: {type(e).__name__}: {e}")
                continue
            tune_s = time.time() - t0

            for seed in seeds:
                if seed not in cache:
                    cache[seed] = load_task(task, seed=seed, max_rows=args.max_rows)
                ds = cache[seed]
                try:
                    out = run_model(model, best_cfg, ds, device, seed)
                except Exception as e:  # noqa: BLE001
                    print(f"  [eval] {model} seed{seed} FAILED: {e}")
                    continue
                test = out["test"]
                sc = test.get("auc", test.get("r2"))
                rows.append({"task_id": task, "dataset": ds.name, "model": model,
                             "seed": seed, "score": sc, "primary": pm,
                             "val_tuned": best_val, **test})
            # Persist after each model so long runs are resumable/inspectable.
            pd.DataFrame(rows).to_csv(long_path, index=False)
            best_line = ", ".join(f"{k}={v}" for k, v in best_cfg.items()
                                  if k not in ("device", "hidden_dims"))
            print(f"  {model:26s} val={best_val:.4f}  ({tune_s:.0f}s tune, "
                  f"{args.trials} trials)")

        print(f"[suite] {ds0.name} done in {time.time()-t_task:.0f}s")

    df = pd.DataFrame(rows)
    df.to_csv(long_path, index=False)
    print(f"\n[suite] long results -> {long_path}")

    # Aggregate (ranks / CD / Wilcoxon on the unified score).
    rep = full_report(df, "score", args.out, reference=args.reference)
    print("\n=== AVERAGE RANK (lower = better) ===")
    print(rep["avg_rank"].to_string())
    if rep["wilcoxon"] is not None:
        print(f"\n=== WILCOXON vs {args.reference} ===")
        print(rep["wilcoxon"].to_string(index=False))
    print(f"\n[suite] CD = {rep['cd']:.3f} | artifacts in {args.out}/")
    with open(os.path.join(args.out, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)


if __name__ == "__main__":
    main()
