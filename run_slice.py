"""Vertical slice: run ONE dataset through every model cell and print a table.

This is the de-risking harness. It answers, on a single dataset, the paper's
core question before we spend compute on the full suite:

  trees (xgb/cat/lgbm/rf)  vs  raw NNs (mlp/tabresnet)  vs
  leaf-onehot->MLP  vs  TKCE two-stage (gbt/mondrian kernel)  vs  TKCE joint

Usage:
    python run_slice.py --task 361065 --max-rows 4000 --device cpu
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd

from tkce.baselines import fit_tree_baseline, leaf_onehot_features
from tkce.data import load_task
from tkce.kernels import build_kernel
from tkce.metrics import primary_metric, score
from tkce.train import (encode, predict_head, predict_joint, pretrain_encoder,
                        resolve_device, train_head, train_joint)


def _nn_cell(kind, Xtr, ytr, Xva, yva, Xte, dataset, cfg, device):
    model = train_head(kind, Xtr, ytr, Xva, yva, dataset, cfg, device)
    proba = predict_head(model, Xte, dataset, device)
    return score(dataset, dataset.y_test, proba)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", type=int, default=361065)   # MagicTelescope
    ap.add_argument("--max-rows", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    cfg = dict(
        seed=args.seed, device=args.device,
        # encoder / contrastive
        embedding_dim=64, enc_hidden=(256, 128), enc_lr=1e-3,
        contrastive_loss="infonce", temperature=0.1, pos_threshold=0.6,
        max_pos=50, pretrain_epochs=25, batch_size=256, weight_decay=1e-4,
        # heads
        hidden_dims=(256, 128), dropout=0.1, head_lr=1e-3, head_epochs=80,
        patience=16, d=192, d_hidden=256, n_blocks=3,
        # joint
        lambda_contrast=0.5,
        # kernels
        gbt=dict(n_estimators=200, max_depth=4, random_state=args.seed),
        mondrian=dict(n_estimators=100, max_depth=6, random_state=args.seed),
    )
    device = resolve_device(args.device)

    print(f"[slice] loading OpenML task {args.task} (cap {args.max_rows} rows)...")
    ds = load_task(args.task, seed=args.seed, max_rows=args.max_rows)
    pm = primary_metric(ds.task_type)
    print(f"[slice] {ds.name} | {ds.task_type} | n={ds.meta['n_rows']} "
          f"num={ds.meta['n_num']} cat={ds.meta['n_cat']} classes={ds.n_classes} "
          f"| feat={ds.n_features} onehot={ds.n_features_oh} | primary metric={pm}")

    rows = []

    def record(name, m, t0):
        m = {**m, "model": name, "seconds": round(time.time() - t0, 1)}
        rows.append(m)
        val = m.get(pm, float("nan"))
        print(f"  {name:32s} {pm}={val:.4f}  ({m['seconds']}s)")

    # --- Tree baselines ---
    print("[slice] tree baselines...")
    for name in ["xgboost", "lightgbm", "catboost", "random_forest"]:
        t0 = time.time()
        test, _val, _model = fit_tree_baseline(name, ds, cfg)
        record(name, test, t0)

    # --- Raw NN baselines (one-hot features) ---
    print("[slice] raw NN baselines...")
    for kind in ["mlp", "tabresnet"]:
        t0 = time.time()
        m = _nn_cell(kind, ds.Xoh_train, ds.y_train, ds.Xoh_val, ds.y_val,
                     ds.Xoh_test, ds, cfg, device)
        record(f"{kind}_raw", m, t0)

    # --- Leaf-one-hot -> MLP (He et al. 2014) ---
    print("[slice] leaf-onehot -> MLP baseline...")
    t0 = time.time()
    Ltr, Lva, Lte = leaf_onehot_features(ds, cfg["gbt"])
    m = _nn_cell("mlp", Ltr, ds.y_train, Lva, ds.y_val, Lte, ds, cfg, device)
    record("leafonehot_mlp", m, t0)

    # --- TKCE two-stage (frozen encoder) for gbt + mondrian kernels ---
    for kname in ["gbt", "mondrian"]:
        print(f"[slice] TKCE two-stage | kernel={kname} ...")
        t0 = time.time()
        kern = build_kernel(kname, ds.X_train, ds.y_train, ds.task_type, cfg[kname])
        enc, hist = pretrain_encoder(
            ds.X_train, kern,
            {**cfg, "contrastive_loss": cfg["contrastive_loss"]}, device)
        emb_tr = encode(enc, ds.X_train, device)
        emb_va = encode(enc, ds.X_val, device)
        emb_te = encode(enc, ds.X_test, device)
        pre_s = round(time.time() - t0, 1)
        print(f"    pretrain done ({pre_s}s, final contrastive loss={hist[-1]:.3f})")
        for kind in ["mlp", "tabresnet"]:
            t1 = time.time()
            m = _nn_cell(kind, emb_tr, ds.y_train, emb_va, ds.y_val, emb_te,
                         ds, cfg, device)
            record(f"tkce2stage_{kname}_{kind}", m, t1)

    # --- TKCE joint (gbt kernel) ---
    print("[slice] TKCE joint (gbt kernel)...")
    kern = build_kernel("gbt", ds.X_train, ds.y_train, ds.task_type, cfg["gbt"])
    for kind in ["mlp"]:
        t0 = time.time()
        model = train_joint(kind, ds.X_train, ds.y_train, ds.X_val, ds.y_val,
                            kern, ds, cfg, device)
        proba = predict_joint(model, ds.X_test, ds, device)
        record(f"tkce_joint_{kind}", score(ds, ds.y_test, proba), t0)

    # --- Summary ---
    df = pd.DataFrame(rows).sort_values(pm, ascending=False)
    cols = ["model"] + [c for c in ["auc", "accuracy", "rmse", "r2"] if c in df.columns] + ["seconds"]
    print("\n" + "=" * 70)
    print(f"RESULTS — {ds.name} ({ds.task_type}), ranked by {pm}")
    print("=" * 70)
    print(df[cols].to_string(index=False))
    out = f"results/slice_{ds.name}_seed{args.seed}.csv"
    df.to_csv(out, index=False)
    print(f"\n[slice] saved -> {out}")


if __name__ == "__main__":
    main()
