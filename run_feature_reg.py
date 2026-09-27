"""Feature regularisation on a convex linear head: binary, multi-class, regression.

The head is FIXED and convex, solved to the global optimum, so nothing here
depends on a learning rate, an epoch count or training dynamics:

    classification  scikit-learn LogisticRegression (multinomial for >2 classes)
    regression      scikit-learn Ridge, with alpha = 1/(2C) so the L2 penalty
                    coefficient relative to the summed loss matches the
                    classification setting (the loss functions differ, so this
                    is the same penalty *coefficient*, not an identical
                    effective strength)

All experimentation happens on the FEATURES: the split bits of a frozen random
forest (a classifier or a regressor, matching the task), under the out-of-bag
honest protocol. Four interventions, which compose:

    --flip-uniform p    flip every bit with probability p
    --flip-ramp P       flip with p(depth) = P * depth/max_depth; roots never flip
    --drop-layers 4,5   delete the deepest layers outright
    --keep-layers 0,1,2 keep only these depths and delete the rest

Every metric is reported on ALL THREE splits (train on clean bits, validation,
test): AUC, accuracy and log-loss for classification (macro one-vs-rest AUC
when there are more than two classes); RMSE, R^2 and MAE on the original target
scale for regression. The overfitting gap is train-minus-validation on the
primary metric (AUC or R^2), with the training term measured on CLEAN bits so
the gap is comparable between arms.

HOW NOISE WORKS HERE. A convex solver sees one fixed matrix, so the equivalent
of per-batch resampled flips is to stack --noise-copies independently flipped
copies of the training rows and fit them together, approximating the loss
averaged over the noise. Flips are applied to the hard 0/1 bits and the OOB
mask is applied AFTER. Validation and test always use clean all-trees bits.

    python run_feature_reg.py --task 361055                     # binary
    python run_feature_reg.py --task 32 --keep-layers 0,1,2     # 10-class
    python run_feature_reg.py --task 361072 --flip-uniform 0.2  # regression
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import log_loss, mean_absolute_error

from run_fusion import (encode_margins, node_depths_and_samples, oob_mask_matrix,
                        split_margins, split_direction_encoding)
from tkce.baselines import fit_tree_baseline
from tkce.data import load_task
from tkce.metrics import primary_metric, score


# --------------------------------------------------------------------------- #
# task-type dispatch
# --------------------------------------------------------------------------- #
def fit_encoder(X, y, args, task_type):
    """The frozen forest that defines the bits: classifier or regressor to match
    the task. Everything downstream (split bits, depths, OOB masks) only reads
    the tree structures, which are identical between the two."""
    cls = RandomForestClassifier if task_type == "classification" else RandomForestRegressor
    rf = cls(n_estimators=args.rf_trees, max_depth=args.rf_depth,
             min_samples_leaf=args.rf_min_leaf, n_jobs=-1, random_state=args.seed)
    return rf.fit(X, y)


def make_head(task_type, C, max_iter, copies=1):
    """The convex head at a FIXED penalty strength.

    Stacking K noisy copies of the training rows multiplies the summed data
    loss by K, which would silently weaken the penalty K-fold relative to the
    clean arm. Dividing C by K (multiplying alpha by K) undoes that: the
    objective becomes (1/2)||w||^2 + C * sum_i E_noise[loss_i], the
    expected-loss formulation, with the SAME penalty per original training row
    as the clean arm. Without this, "C held constant" would not be true.

    Logistic: (1/2)||w||^2 + C*sum(loss)  <=>  sum(loss) + (1/(2C))||w||^2
    Ridge:    sum((y-Xw)^2) + alpha*||w||^2   =>  alpha = 1/(2C)
    """
    if task_type == "classification":
        return LogisticRegression(C=C / copies, max_iter=max_iter, n_jobs=-1)
    return Ridge(alpha=copies / (2.0 * C))


def predict(model, X, task_type):
    return model.predict_proba(X) if task_type == "classification" else model.predict(X)


def all_metrics(ds, y, out):
    """Primary metrics from tkce.metrics plus one calibration/scale metric."""
    m = score(ds, y, out)
    if ds.task_type == "classification":
        m["logloss"] = float(log_loss(y, out, labels=list(range(ds.n_classes))))
    else:
        yt, yp = y * ds.y_std + ds.y_mean, out * ds.y_std + ds.y_mean
        m["mae"] = float(mean_absolute_error(yt, yp))
    return m


# --------------------------------------------------------------------------- #
# feature interventions
# --------------------------------------------------------------------------- #
def build_view(view, X, T):
    if view == "x":
        return X
    if view == "tree":
        return T
    return np.concatenate([X, T], axis=1)


def flip_probabilities(depths, args, kept):
    """Per-bit flip probability. A deleted bit is forced to zero so a flip can
    never resurrect a column that was removed on purpose."""
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
    """Stack `copies` independently flipped copies of the hard training bits.
    Flip first, then apply the OOB mask -- the same order run_fusion.py uses."""
    out = []
    for _ in range(copies):
        b = hard_bits.copy()
        flips = rng.random(b.shape, dtype=np.float32) < pvec[None, :]
        b[flips] = 1.0 - b[flips]
        if maskscale is not None:
            b *= maskscale
        out.append(b)
    return np.concatenate(out, axis=0) if len(out) > 1 else out[0]


# --------------------------------------------------------------------------- #
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
                    help="semicolon-separated inputs; default 'tree' = bits only so "
                         "an unperturbed raw-feature shortcut cannot mask the noise")
    ap.add_argument("--C", type=float, default=0.01,
                    help="inverse regularisation strength, held CONSTANT across arms "
                         "(regression uses Ridge alpha = 1/(2C))")
    ap.add_argument("--max-iter", type=int, default=2000)
    ap.add_argument("--flip-uniform", type=float, default=0.0)
    ap.add_argument("--flip-ramp", type=float, default=0.0)
    ap.add_argument("--flip-ramp-rel", action="store_true")
    ap.add_argument("--drop-layers", type=str, default=None)
    ap.add_argument("--keep-layers", type=str, default=None)
    ap.add_argument("--noise-copies", type=int, default=3)
    ap.add_argument("--no-copy-correction", action="store_true",
                    help="do NOT rescale the penalty by the number of noisy copies "
                         "(reproduces the earlier credit run, where noise arms were "
                         "effectively fitted at C x copies)")
    ap.add_argument("--repeats", type=int, default=1,
                    help="refit with fresh noise this many times; reports the spread")
    ap.add_argument("--label", default=None,
                    help="short arm name written into the JSON (for suite tables)")
    ap.add_argument("--out", default="results/feature_reg")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    if args.flip_uniform > 0 and args.flip_ramp > 0:
        raise SystemExit("pick one of --flip-uniform / --flip-ramp")
    if args.drop_layers and args.keep_layers:
        raise SystemExit("pick one of --drop-layers / --keep-layers")

    ds = load_task(args.task, seed=args.seed, max_rows=args.max_rows)
    tt, prim = ds.task_type, primary_metric(ds.task_type)
    head_desc = (f"LogisticRegression, C={args.C:g}"
                 + (f", multinomial over {ds.n_classes} classes" if ds.n_classes > 2 else "")
                 if tt == "classification" else
                 f"Ridge, alpha=1/(2C)={1/(2*args.C):g}  (C={args.C:g})")
    print(f"\n=== FEATURE REGULARISATION: {ds.name} (task {args.task}) | {tt}"
          f"{f', {ds.n_classes} classes' if tt == 'classification' else ''} ===")
    print(f"[head] {head_desc}, max_iter={args.max_iter}; primary metric = {prim}",
          flush=True)
    print(f"[data] train={len(ds.y_train)} val={len(ds.y_val)} test={len(ds.y_test)} "
          f"| {ds.X_train.shape[1]} raw features", flush=True)

    # ---- the encoding ----
    rf = fit_encoder(ds.X_train, ds.y_train, args, tt)
    hard_train = encode_margins(split_margins(rf, ds.X_train), 0.0).astype(np.float32)
    T_val = split_direction_encoding(rf, ds.X_val).astype(np.float32)
    T_test = split_direction_encoding(rf, ds.X_test).astype(np.float32)
    depths, samples = node_depths_and_samples(rf)
    n_bits = len(depths)
    print(f"[bits] {n_bits} split bits from {args.rf_trees} "
          f"{'classification' if tt == 'classification' else 'regression'} trees of "
          f"depth {args.rf_depth}; bits per depth "
          f"{ {int(d): int((depths == d).sum()) for d in np.unique(depths)} }", flush=True)

    clean_train, maskscale = hard_train, None
    if args.encoding == "oob":
        maskscale, mean_oob = oob_mask_matrix(rf, len(ds.y_train))
        clean_train = hard_train * maskscale
        print(f"[bits] OOB-honest: each training row keeps bits from "
              f"{mean_oob:.1f}/{args.rf_trees} trees (val and test keep all trees)",
              flush=True)

    # ---- which bits survive ----
    kept, what = np.ones(n_bits, dtype=bool), "all layers kept"
    if args.keep_layers:
        keep = [int(v) for v in args.keep_layers.split(",")]
        kept, what = np.isin(depths, keep), f"keep only depths {keep}"
    elif args.drop_layers:
        drop = [int(v) for v in args.drop_layers.split(",")]
        kept, what = ~np.isin(depths, drop), f"drop depths {drop}"
    if not kept.all():
        print(f"[bits] {what}: keeping {int(kept.sum())}/{n_bits} bits at depths "
              f"{sorted(set(depths[kept].tolist()))}; median training rows per split "
              f"{int(np.median(samples[kept]))} kept vs "
              f"{int(np.median(samples[~kept]))} dropped", flush=True)

    # ---- how much each bit is flipped ----
    pvec, mode = flip_probabilities(depths, args, kept)
    per_depth = None
    if pvec is not None:
        per_depth = {int(d): round(float(pvec[depths == d].mean()), 4)
                     for d in np.unique(depths)}
        print(f"[flip] {mode}; hard bits flipped, OOB mask after; "
              f"{args.noise_copies} stacked noisy copies", flush=True)
        print(f"[flip] mean p by depth {per_depth}; mean over all bits "
              f"{pvec.mean():.4f}", flush=True)
    else:
        print("[flip] no noise (clean arm)", flush=True)

    # ---- reference ceiling: the tree baselines on raw features ----
    print("[ceiling] fitting tree baselines ...", flush=True)
    ceil = {}
    for name in ("xgboost", "lightgbm", "random_forest"):
        te, _, _ = fit_tree_baseline(name, ds, {"seed": args.seed})
        ceil[name] = te
        print(f"  {name:14s} " + "  ".join(f"{k}={v:.4f}" for k, v in te.items()), flush=True)
    best_tree = max(ceil, key=lambda k: ceil[k][prim])
    tree_ceiling = ceil[best_tree][prim]

    # ---- fit ----
    results = []
    for view in [v.strip() for v in args.views.split(";") if v.strip()]:
        runs = []
        for rep in range(args.repeats):
            rng = np.random.default_rng(args.seed + 100 * rep)
            t0 = time.time()
            if pvec is None:
                bits_tr, y_tr, copies = clean_train, ds.y_train, 1
            else:
                copies = args.noise_copies
                bits_tr = noisy_training_matrix(hard_train, maskscale, pvec, copies, rng)
                y_tr = np.tile(ds.y_train, copies)
            Xtr = build_view(view, np.tile(ds.X_train, (copies, 1)), bits_tr[:, kept])
            Xtr_clean = build_view(view, ds.X_train, clean_train[:, kept])
            Xva = build_view(view, ds.X_val, T_val[:, kept])
            Xte = build_view(view, ds.X_test, T_test[:, kept])
            head_copies = 1 if args.no_copy_correction else copies
            model = make_head(tt, args.C, args.max_iter, head_copies).fit(Xtr, y_tr)
            if copies > 1 and rep == 0 and view == "tree":
                eff = (f"C/{head_copies} = {args.C / head_copies:g}" if tt == "classification"
                       else f"alpha x {head_copies} = {head_copies / (2 * args.C):g}")
                print(f"[head] {copies} stacked copies -> penalty rescaled so the strength "
                      f"per ORIGINAL row equals the clean arm ({eff})"
                      + ("  [correction DISABLED]" if args.no_copy_correction else ""),
                      flush=True)
            # Train metrics on CLEAN bits, so the gap is comparable across arms.
            r = dict(train=all_metrics(ds, ds.y_train, predict(model, Xtr_clean, tt)),
                     val=all_metrics(ds, ds.y_val, predict(model, Xva, tt)),
                     test=all_metrics(ds, ds.y_test, predict(model, Xte, tt)),
                     n_features=int(Xtr.shape[1]), n_rows=int(Xtr.shape[0]),
                     seconds=round(time.time() - t0, 1))
            r["gap"] = r["train"][prim] - r["val"][prim]
            runs.append(r)
            tag = f" rep {rep + 1}/{args.repeats}" if args.repeats > 1 else ""
            print(f"\n[{view}]{tag} {r['n_features']} features, {r['n_rows']} training rows"
                  f"{f' ({copies} noisy copies)' if copies > 1 else ''}   ({r['seconds']}s)",
                  flush=True)
            keys = list(r["train"].keys())
            print("  " + f"{'split':6s}" + "".join(f"{k:>10s}" for k in keys))
            for split in ("train", "val", "test"):
                print("  " + f"{split:6s}" + "".join(f"{r[split][k]:>10.4f}" for k in keys)
                      + ("   <- on clean bits" if split == "train" else ""), flush=True)
        agg = dict(view=view, repeats=args.repeats, n_features=runs[0]["n_features"],
                   n_rows=runs[0]["n_rows"], seconds=runs[0]["seconds"])
        for split in ("train", "val", "test"):
            agg[split] = {k: float(np.mean([rr[split][k] for rr in runs]))
                          for k in runs[0][split]}
        agg["gap"] = float(np.mean([rr["gap"] for rr in runs]))
        agg["test_primary_std"] = float(np.std([rr["test"][prim] for rr in runs]))
        agg["per_repeat_test"] = [round(rr["test"][prim], 4) for rr in runs]
        results.append(agg)
        if args.repeats > 1:
            print(f"  -> {view}: test {prim} {agg['test'][prim]:.4f} "
                  f"+/- {agg['test_primary_std']:.4f} over {args.repeats} noise draws")

    # ---- verdict ----
    print("\n" + "=" * 70)
    print(f"{ds.name} | {tt} | {head_desc} | {what} | {mode}")
    print("=" * 70)
    print(f"  {'view':8s} {'features':>9s} {'train':>9s} {'val':>9s} {'test':>9s} {'gap':>9s}"
          f"   ({prim})")
    for r in results:
        print(f"  {r['view']:8s} {r['n_features']:9d} {r['train'][prim]:9.4f} "
              f"{r['val'][prim]:9.4f} {r['test'][prim]:9.4f} {r['gap']:+9.4f}")
    print(f"  {'best tree':8s} {'':9s} {'':9s} {'':9s} {tree_ceiling:9.4f}"
          f"           ({best_tree})")
    print(f"\n  train-minus-validation gap on {prim}, train measured on CLEAN bits")

    summary = dict(dataset=ds.name, task=args.task, task_type=tt, n_classes=ds.n_classes,
                   primary=prim, label=args.label, seed=args.seed, C=args.C,
                   head=head_desc, encoding=args.encoding, rf_trees=args.rf_trees,
                   rf_depth=args.rf_depth, n_raw_features=int(ds.X_train.shape[1]),
                   n_train=int(len(ds.y_train)), n_val=int(len(ds.y_val)),
                   n_test=int(len(ds.y_test)), n_bits_total=n_bits,
                   kept_n_bits=int(kept.sum()),
                   kept_depths=sorted(set(depths[kept].tolist())), bits_selection=what,
                   flip_uniform=args.flip_uniform, flip_ramp=args.flip_ramp,
                   flip_ramp_rel=bool(args.flip_ramp_rel), flip_mode=mode,
                   flip_mean_p=float(pvec.mean()) if pvec is not None else 0.0,
                   flip_per_depth=per_depth, noise_copies=args.noise_copies,
                   copy_correction=not args.no_copy_correction,
                   repeats=args.repeats, ceiling=ceil, best_tree=best_tree,
                   tree_ceiling=tree_ceiling, results=results)
    path = os.path.join(args.out, f"featreg_{ds.name}.json")
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[featreg] summary -> {path}")


if __name__ == "__main__":
    main()
