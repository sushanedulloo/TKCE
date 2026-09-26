"""Feature regularisation on a scikit-learn logistic regression.

The head is FIXED: sklearn's LogisticRegression, one weight per bit, no hidden
layer, no nonlinearity, convex objective solved to the global optimum. Nothing
here depends on a learning rate, an epoch count or any training dynamics. All
experimentation happens on the FEATURES.

Four things can be done to the tree encoding, and they compose:

    --flip-uniform p    flip every bit with probability p
    --flip-ramp P       flip with p(depth) = P * depth/max_depth, so the root
                        splits are never touched and the deepest layer gets P
    --drop-layers 4,5   delete the last few layers outright
    --keep-layers 0,1,2 keep only these depths and delete the rest

HOW NOISE WORKS HERE, AND WHY IT DIFFERS FROM THE SGD VERSION
A mini-batch trainer can draw fresh flips every batch, which makes the noise a
regulariser rather than one corrupted dataset. A convex solver sees one fixed
matrix, so the equivalent is to stack SEVERAL independently flipped copies of
the training rows and fit on all of them at once: that approximates minimising
the loss AVERAGED over the noise, which is what per-batch resampling does.
--noise-copies sets how many copies. One copy is a single corrupted dataset and
is much noisier as an estimate; more copies cost memory and time linearly.

Flips are applied to the HARD 0/1 bits and the OOB-honesty mask is applied
AFTER, matching run_fusion.py exactly. Validation and test always use the clean
all-trees encoding, never a flipped one.

    python run_feature_reg.py --task 361055                       # plain baseline
    python run_feature_reg.py --task 361055 --flip-uniform 0.1
    python run_feature_reg.py --task 361055 --drop-layers 5 --flip-ramp 0.2
    python run_feature_reg.py --task 361055 --keep-layers 0,1,2
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from run_fusion import (encode_margins, fit_encoder_rf, node_depths_and_samples,
                        oob_mask_matrix, split_margins, split_direction_encoding)
from tkce.baselines import fit_tree_baseline
from tkce.data import load_task


def build_view(view, X, T):
    """Assemble the input matrix for one view. 'tree' = bits only."""
    if view == "x":
        return X
    if view == "tree":
        return T
    return np.concatenate([X, T], axis=1)


def flip_probabilities(depths, args, kept):
    """Per-bit flip probability, shape (n_bits,). Zero everywhere means no noise.

    A deleted bit never flips: its probability is forced to zero so a flip can
    never resurrect a column we removed on purpose.
    """
    if args.flip_uniform > 0:
        p = np.full(len(depths), args.flip_uniform, dtype=np.float32)
        mode = f"UNIFORM p={args.flip_uniform:g} on every bit"
    elif args.flip_ramp > 0:
        denom = int(depths[kept].max()) if args.flip_ramp_rel else int(depths.max())
        denom = max(1, denom)
        p = (args.flip_ramp * depths / denom).astype(np.float32)
        mode = (f"RAMP p(depth)={args.flip_ramp:g} x depth/{denom}"
                + ("  (relative to the deepest surviving layer)"
                   if args.flip_ramp_rel else ""))
    else:
        return None, "no noise"
    p = p.copy()
    p[~kept] = 0.0
    return p, mode


def noisy_training_matrix(hard_bits, maskscale, pvec, copies, rng):
    """Stack `copies` independently flipped copies of the training bits.

    hard_bits : (n_rows, n_bits) of 0/1, BEFORE the OOB mask
    maskscale : (n_rows, n_bits) OOB multiplier, or None
    Returns (n_rows * copies, n_bits). Flip first, then mask — the same order
    run_fusion.py uses, so the two implementations stay comparable.
    """
    out = []
    for _ in range(copies):
        b = hard_bits.copy()
        flips = rng.random(b.shape, dtype=np.float32) < pvec[None, :]
        b[flips] = 1.0 - b[flips]
        if maskscale is not None:
            b *= maskscale
        out.append(b)
    return np.concatenate(out, axis=0) if len(out) > 1 else out[0]


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
    ap.add_argument("--views", default="tree",
                    help="semicolon-separated inputs; default 'tree' = bits only, "
                         "so an unperturbed raw-feature shortcut cannot mask the "
                         "effect of the noise")
    # ---- the head: fixed, and deliberately not tuned per arm ----
    ap.add_argument("--C", type=float, default=0.01,
                    help="inverse regularisation strength, held CONSTANT across "
                         "every arm so that any difference in score is caused by "
                         "the features and not by a differently tuned penalty")
    ap.add_argument("--max-iter", type=int, default=2000)
    # ---- what we do to the features ----
    ap.add_argument("--flip-uniform", type=float, default=0.0)
    ap.add_argument("--flip-ramp", type=float, default=0.0,
                    help="p(depth) = P * depth/max_depth; roots never flip")
    ap.add_argument("--flip-ramp-rel", action="store_true",
                    help="scale the ramp against the deepest SURVIVING layer, so "
                         "the full P is reached even after deleting layers")
    ap.add_argument("--drop-layers", type=str, default=None)
    ap.add_argument("--keep-layers", type=str, default=None)
    ap.add_argument("--noise-copies", type=int, default=3,
                    help="independently flipped copies of the training set, stacked "
                         "and fitted together, approximating the loss averaged over "
                         "the noise. Memory and time grow linearly; 1 copy is a "
                         "single corrupted dataset and a much noisier estimate")
    ap.add_argument("--repeats", type=int, default=1,
                    help="refit with fresh noise this many times and report "
                         "mean +/- spread, since the flips are random")
    ap.add_argument("--out", default="results/feature_reg")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    if args.flip_uniform > 0 and args.flip_ramp > 0:
        raise SystemExit("pick one of --flip-uniform / --flip-ramp")
    if args.drop_layers and args.keep_layers:
        raise SystemExit("pick one of --drop-layers / --keep-layers")

    ds = load_task(args.task, seed=args.seed, max_rows=args.max_rows)
    print(f"\n=== FEATURE REGULARISATION: {ds.name} (task {args.task}) ===")
    print(f"[head] sklearn LogisticRegression, C={args.C:g} (held constant), "
          f"max_iter={args.max_iter}", flush=True)
    print(f"[data] train={len(ds.y_train)} val={len(ds.y_val)} "
          f"test={len(ds.y_test)}", flush=True)

    # ---- the encoding, exactly as run_fusion builds it ----
    rf = fit_encoder_rf(ds.X_train, ds.y_train, args)
    hard_train = encode_margins(split_margins(rf, ds.X_train), 0.0).astype(np.float32)
    T_val = split_direction_encoding(rf, ds.X_val).astype(np.float32)
    T_test = split_direction_encoding(rf, ds.X_test).astype(np.float32)
    depths, samples = node_depths_and_samples(rf)
    n_bits = len(depths)
    print(f"[bits] {n_bits} split bits from {args.rf_trees} trees of depth "
          f"{args.rf_depth}; bits per depth "
          f"{ {int(d): int((depths == d).sum()) for d in np.unique(depths)} }",
          flush=True)

    # The clean training encoding: what the model is judged on, never flipped.
    clean_train = hard_train
    maskscale = None
    if args.encoding == "oob":
        maskscale, mean_oob = oob_mask_matrix(rf, len(ds.y_train))
        clean_train = hard_train * maskscale
        print(f"[bits] OOB-honest: each training row keeps bits from "
              f"{mean_oob:.1f}/{args.rf_trees} trees "
              f"(val and test keep all trees)", flush=True)

    # ---- which bits survive ----
    kept = np.ones(n_bits, dtype=bool)
    what = "all layers kept"
    if args.keep_layers:
        keep = [int(v) for v in args.keep_layers.split(",")]
        kept, what = np.isin(depths, keep), f"keep only depths {keep}"
    elif args.drop_layers:
        drop = [int(v) for v in args.drop_layers.split(",")]
        kept, what = ~np.isin(depths, drop), f"drop depths {drop}"
    if not kept.all():
        print(f"[bits] {what}: keeping {int(kept.sum())}/{n_bits} bits at depths "
              f"{sorted(set(depths[kept].tolist()))}; median training rows per "
              f"split {int(np.median(samples[kept]))} kept vs "
              f"{int(np.median(samples[~kept]))} dropped", flush=True)

    # ---- how much each bit gets flipped ----
    pvec, mode = flip_probabilities(depths, args, kept)
    if pvec is not None:
        per_depth = {int(d): round(float(pvec[depths == d].mean()), 4)
                     for d in np.unique(depths)}
        print(f"[flip] {mode}; applied to the hard bits, OOB mask after; "
              f"{args.noise_copies} stacked noisy copies", flush=True)
        print(f"[flip] mean p by depth {per_depth}", flush=True)
        print(f"[flip] {int((pvec > 0).sum())} bits can flip; mean p over all "
              f"{n_bits} bits = {pvec.mean():.4f}  <- a matched uniform control "
              f"uses this number", flush=True)
    else:
        per_depth = None
        print("[flip] no noise (this is a clean arm)", flush=True)

    # ---- reference ceiling ----
    print("[ceiling] fitting tree baselines ...", flush=True)
    ceil = {}
    for name in ("xgboost", "lightgbm", "random_forest"):
        te, _, _ = fit_tree_baseline(name, ds, {"seed": args.seed})
        ceil[name] = te["auc"]
        print(f"  {name:14s} AUC={te['auc']:.4f}", flush=True)
    tree_ceiling = max(ceil.values())

    # ---- fit ----
    results = []
    for view in [v.strip() for v in args.views.split(";") if v.strip()]:
        runs = []
        for rep in range(args.repeats):
            rng = np.random.default_rng(args.seed + 100 * rep)
            t0 = time.time()
            if pvec is None:
                bits_tr = clean_train
                y_tr = ds.y_train
                copies = 1
            else:
                copies = args.noise_copies
                bits_tr = noisy_training_matrix(hard_train, maskscale, pvec,
                                                copies, rng)
                y_tr = np.tile(ds.y_train, copies)
            # Deleted bits are dropped as columns, so the reported feature count
            # is what the model actually sees.
            Xtr = build_view(view, np.tile(ds.X_train, (copies, 1)), bits_tr[:, kept])
            Xva = build_view(view, ds.X_val, T_val[:, kept])
            Xte = build_view(view, ds.X_test, T_test[:, kept])
            lr = LogisticRegression(C=args.C, max_iter=args.max_iter, n_jobs=-1)
            lr.fit(Xtr, y_tr)
            # The overfitting gap must be measured on the CLEAN training bits, not
            # on the flipped matrix the model was fitted to. Otherwise a noisy arm
            # looks like it overfits more purely because its training inputs were
            # corrupted, and the gap stops being comparable between arms.
            Xtr_clean = build_view(view, ds.X_train, clean_train[:, kept])
            r = dict(train_auc=roc_auc_score(
                         ds.y_train, lr.predict_proba(Xtr_clean)[:, 1]),
                     train_auc_on_fit=roc_auc_score(
                         y_tr, lr.predict_proba(Xtr)[:, 1]),
                     val_auc=roc_auc_score(ds.y_val, lr.predict_proba(Xva)[:, 1]),
                     test_auc=roc_auc_score(ds.y_test, lr.predict_proba(Xte)[:, 1]),
                     n_features=int(Xtr.shape[1]), n_rows=int(Xtr.shape[0]),
                     seconds=round(time.time() - t0, 1))
            runs.append(r)
            tag = f" rep {rep + 1}/{args.repeats}" if args.repeats > 1 else ""
            print(f"\n[{view}]{tag} {r['n_features']} features, "
                  f"{r['n_rows']} training rows"
                  f"{f' ({copies} noisy copies)' if copies > 1 else ''}", flush=True)
            print(f"  train_auc={r['train_auc']:.4f} (on clean bits)  "
                  f"val_auc={r['val_auc']:.4f}  test_auc={r['test_auc']:.4f}"
                  f"   ({r['seconds']}s)", flush=True)
        agg = dict(view=view, repeats=args.repeats,
                   n_features=runs[0]["n_features"], n_rows=runs[0]["n_rows"],
                   train_auc=float(np.mean([r["train_auc"] for r in runs])),
                   train_auc_on_fit=float(np.mean(
                       [r["train_auc_on_fit"] for r in runs])),
                   val_auc=float(np.mean([r["val_auc"] for r in runs])),
                   test_auc=float(np.mean([r["test_auc"] for r in runs])),
                   test_auc_std=float(np.std([r["test_auc"] for r in runs])),
                   per_repeat=[round(r["test_auc"], 4) for r in runs])
        results.append(agg)
        if args.repeats > 1:
            print(f"  -> {view}: test {agg['test_auc']:.4f} "
                  f"+/- {agg['test_auc_std']:.4f} over {args.repeats} noise draws",
                  flush=True)

    # ---- verdict ----
    print("\n" + "=" * 68)
    print(f"{ds.name} | logistic regression, C={args.C:g} | {what}"
          f" | {mode}")
    print("=" * 68)
    print(f"  {'view':9s} {'features':>9s} {'train':>8s} {'val':>8s} {'test':>8s}")
    for r in results:
        print(f"  {r['view']:9s} {r['n_features']:9d} {r['train_auc']:8.4f} "
              f"{r['val_auc']:8.4f} {r['test_auc']:8.4f}")
    print(f"  {'best tree':9s} {'':9s} {'':8s} {'':8s} {tree_ceiling:8.4f}")
    gap = results[0]["train_auc"] - results[0]["val_auc"]
    print(f"\n  train-minus-validation gap: {gap:+.4f}  "
          f"<- how much this arm overfits (train measured on CLEAN bits, so "
          f"this is comparable across arms)")

    summary = dict(dataset=ds.name, task=args.task, seed=args.seed, C=args.C,
                   encoding=args.encoding, rf_trees=args.rf_trees,
                   rf_depth=args.rf_depth, n_bits_total=n_bits,
                   kept_n_bits=int(kept.sum()),
                   kept_depths=sorted(set(depths[kept].tolist())),
                   bits_selection=what,
                   flip_uniform=args.flip_uniform, flip_ramp=args.flip_ramp,
                   flip_ramp_rel=bool(args.flip_ramp_rel),
                   flip_mode=mode, flip_mean_p=float(pvec.mean()) if pvec is not None else 0.0,
                   flip_per_depth=per_depth, noise_copies=args.noise_copies,
                   repeats=args.repeats, ceiling=ceil, tree_ceiling=tree_ceiling,
                   results=results)
    path = os.path.join(args.out, f"featreg_{ds.name}.json")
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[featreg] summary -> {path}")


if __name__ == "__main__":
    main()
