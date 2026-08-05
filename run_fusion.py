"""Three-view feature-fusion model (the professor's variation) on ONE dataset.

The pipeline collapses the contrastive machinery into a plain SUPERVISED network
that sees the same row x through three complementary "views", concatenates them,
and classifies:

    view 1  x                      raw tabular features
    view 2  tree encoding          Random-Forest SPLIT-DIRECTION bits:
                                    for every internal split node, 1 = the sample
                                    goes LEFT, 0 = goes RIGHT (the leaf is captured
                                    implicitly by the path of directions)
    view 3  tabresnet(x)           a TabResNet used as a FEATURE EXTRACTOR on x
                                    (its learned representation, trained jointly)

    concat[ x || tree_enc || tabresnet(x) ]  ->  fusion net (TabResNet or MLP)
                                             ->  prediction head  ->  y_hat

No contrastive loss, no Siamese. Everything except the (frozen) Random Forest is
trained end-to-end with cross-entropy. The RF is fit once on the train split and
the split-direction encoding is precomputed for train/val/test.

Runs, for comparison:
  * tree ceiling            xgboost / lightgbm / random_forest
  * raw baseline            fusion net on x only  (no tree, no deep view)
  * FULL                    x + tree + tabresnet(x)
  * (--ablation)            also x+tree and x+deep to see which view helps

Colab (GPU):
  !python -u run_fusion.py --task 361070 --epochs 800 --fusion tabresnet --device auto
  # add --ablation to also run the x+tree and x+deep ablations
"""

from __future__ import annotations

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.ensemble import RandomForestClassifier
from torch.utils.data import DataLoader, TensorDataset

from tkce.baselines import fit_tree_baseline
from tkce.data import load_task
from tkce.metrics import clf_metrics
from tkce.models import TabResNet, build_head
from tkce.train import resolve_device


# --------------------------------------------------------------------------- #
# view 2 — Random-Forest split-direction encoding
# --------------------------------------------------------------------------- #
def fit_encoder_rf(X, y, args):
    rf = RandomForestClassifier(
        n_estimators=args.rf_trees, max_depth=args.rf_depth,
        min_samples_leaf=args.rf_min_leaf, n_jobs=-1, random_state=args.seed)
    rf.fit(X, y)
    return rf


def split_direction_encoding(rf, X):
    """For every internal split node across all trees: 1 if the sample would go
    to the LEFT child (X[:, feature] <= threshold), else 0. Concatenated over
    trees -> a dense binary matrix (n_samples, total_internal_nodes)."""
    X = np.ascontiguousarray(X)
    cols = []
    for est in rf.estimators_:
        t = est.tree_
        internal = np.where(t.feature >= 0)[0]          # split nodes only
        feats = t.feature[internal]
        thr = t.threshold[internal].astype(np.float32)
        bits = (X[:, feats] <= thr).astype(np.float32)  # (n, n_internal)
        cols.append(bits)
    return np.concatenate(cols, axis=1) if cols else np.zeros((len(X), 0), np.float32)


# --------------------------------------------------------------------------- #
# the fusion model
# --------------------------------------------------------------------------- #
class FusionModel(nn.Module):
    """Concatenate the selected views, then a fusion net + prediction head.

    views ⊆ {"x", "tree", "deep"}:
      x     -> raw features passed straight in
      tree  -> the RF split-direction encoding
      deep  -> tabresnet(x), a TabResNet feature extractor on x (trained jointly)
    """

    def __init__(self, n_feat, tree_dim, n_classes, views, args):
        super().__init__()
        self.views = views
        cat_dim = 0
        if "x" in views:
            cat_dim += n_feat
        if "tree" in views:
            cat_dim += tree_dim
        self.extractor = None
        if "deep" in views:
            self.extractor = TabResNet(n_feat, args.feat_dim, d=args.ext_d,
                                       d_hidden=args.ext_d_hidden,
                                       n_blocks=args.ext_blocks, dropout=args.dropout)
            cat_dim += args.feat_dim
        fcfg = dict(d=args.d, d_hidden=args.d_hidden, n_blocks=args.n_blocks,
                    dropout=args.dropout, hidden_dims=(args.d, args.d // 2))
        self.fusion = build_head(args.fusion, cat_dim, n_classes, fcfg)
        self.cat_dim = cat_dim

    def forward(self, x, tenc):
        parts = []
        if "x" in self.views:
            parts.append(x)
        if "tree" in self.views:
            parts.append(tenc)
        if self.extractor is not None:
            parts.append(self.extractor(x))
        return self.fusion(torch.cat(parts, dim=1))


# --------------------------------------------------------------------------- #
# train / eval
# --------------------------------------------------------------------------- #
@torch.no_grad()
def evaluate(model, X, T, y, device, bs=8192):
    model.eval()
    ps = []
    for s in range(0, len(X), bs):
        xb = torch.from_numpy(X[s:s + bs]).to(device)
        tb = torch.from_numpy(T[s:s + bs]).to(device)
        ps.append(F.softmax(model(xb, tb), dim=1).cpu().numpy())
    return clf_metrics(y, np.concatenate(ps, axis=0))


def train_model(views, ds, enc, n_classes, args, device, label):
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    model = FusionModel(ds.n_features, enc["train"].shape[1], n_classes,
                        views, args).to(device)
    n_par = sum(p.numel() for p in model.parameters())
    print(f"\n[{label}] views={views}  concat_dim={model.cat_dim}  params={n_par/1e6:.2f}M",
          flush=True)

    loader = DataLoader(
        TensorDataset(torch.from_numpy(ds.X_train).to(device),
                      torch.from_numpy(enc["train"]).to(device),
                      torch.from_numpy(ds.y_train).to(device)),
        batch_size=args.batch_size, shuffle=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay)

    best_auc, best_state, best_ep = -1.0, None, 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        for xb, tb, yb in loader:
            opt.zero_grad()
            loss = F.cross_entropy(model(xb, tb), yb)
            loss.backward(); opt.step()
        va = evaluate(model, ds.X_val, enc["val"], ds.y_val, device)
        if va["auc"] > best_auc:
            best_auc, best_ep = va["auc"], epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if epoch % max(1, args.epochs // 8) == 0:
            print(f"    epoch {epoch:4d}/{args.epochs}  val_auc={va['auc']:.4f}  "
                  f"(best {best_auc:.4f} @ {best_ep})", flush=True)

    model.load_state_dict(best_state)
    te = evaluate(model, ds.X_test, enc["test"], ds.y_test, device)
    print(f"  -> {label:22s} TEST auc={te['auc']:.4f} acc={te['accuracy']:.4f} "
          f"(best val {best_auc:.4f} @ epoch {best_ep})", flush=True)
    return dict(model=label, views="+".join(views), test_auc=te["auc"],
                test_acc=te["accuracy"], best_val_auc=best_auc, best_epoch=best_ep,
                concat_dim=model.cat_dim, params_M=round(n_par / 1e6, 2))


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", type=int, default=361070)          # eye_movements
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-rows", type=int, default=16000)
    # RF encoder (view 2) — depth is capped so the encoding width stays sane
    ap.add_argument("--rf-trees", type=int, default=100)
    ap.add_argument("--rf-depth", type=int, default=6,
                    help="max depth of the encoding RF; controls the encoding width")
    ap.add_argument("--rf-min-leaf", type=int, default=5)
    # deep view (view 3) extractor
    ap.add_argument("--feat-dim", type=int, default=128, help="tabresnet(x) output width")
    ap.add_argument("--ext-d", type=int, default=192)
    ap.add_argument("--ext-d-hidden", type=int, default=256)
    ap.add_argument("--ext-blocks", type=int, default=3)
    # fusion net + head
    ap.add_argument("--fusion", default="tabresnet", choices=["tabresnet", "mlp"])
    ap.add_argument("--d", type=int, default=256)
    ap.add_argument("--d-hidden", type=int, default=512)
    ap.add_argument("--n-blocks", type=int, default=4)
    ap.add_argument("--dropout", type=float, default=0.1)
    # optimisation
    ap.add_argument("--epochs", type=int, default=800)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    # misc
    ap.add_argument("--ablation", action="store_true",
                    help="also run x+tree and x+deep to isolate each view")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", default="results/fusion")
    args = ap.parse_args()

    device = resolve_device(args.device)
    os.makedirs(args.out, exist_ok=True)
    ds = load_task(args.task, seed=args.seed, max_rows=args.max_rows)
    if ds.task_type != "classification":
        raise SystemExit(f"{ds.name} is not classification.")
    C = ds.n_classes
    print(f"\n=== FUSION: {ds.name} (task {args.task}) | {ds.n_features} feats, "
          f"{C} classes | fusion={args.fusion} | device={device} ===", flush=True)

    # -------- view 2: RF split-direction encoding --------
    print(f"[view2] fitting encoding RF ({args.rf_trees} trees, depth {args.rf_depth}) ...",
          flush=True)
    rf = fit_encoder_rf(ds.X_train, ds.y_train, args)
    enc = {"train": split_direction_encoding(rf, ds.X_train),
           "val":   split_direction_encoding(rf, ds.X_val),
           "test":  split_direction_encoding(rf, ds.X_test)}
    E = enc["train"].shape[1]
    print(f"[view2] tree-encoding width = {E} split bits "
          f"(~{E * enc['train'].shape[0] * 4 / 1e6:.0f} MB train)", flush=True)

    # -------- tree ceiling (reference) --------
    print("[ceiling] fitting tree baselines ...", flush=True)
    ceil = {}
    for name in ("xgboost", "lightgbm", "random_forest"):
        te, _, _ = fit_tree_baseline(name, ds, {"seed": args.seed})
        ceil[name] = te["auc"]
        print(f"  {name:14s} AUC={te['auc']:.4f}", flush=True)
    tree_ceiling = max(ceil.values())

    # -------- neural runs --------
    if args.ablation:
        run_views = [["x"], ["x", "tree"], ["x", "deep"], ["x", "tree", "deep"]]
    else:
        run_views = [["x"], ["x", "tree", "deep"]]
    labels = {"x": "raw (x only)", "x+tree": "x + tree", "x+deep": "x + deep",
              "x+tree+deep": "FULL (x+tree+deep)"}
    results = []
    for v in run_views:
        lab = labels["+".join(v)]
        results.append(train_model(v, ds, enc, C, args, device, lab))

    # -------- summary --------
    print("\n" + "=" * 64)
    print(f"RESULTS — {ds.name} | fusion={args.fusion} | epochs={args.epochs}")
    print("=" * 64)
    print(f"  {'tree ceiling':22s} {tree_ceiling:.4f}   "
          f"(xgb {ceil['xgboost']:.3f} / lgb {ceil['lightgbm']:.3f} / rf {ceil['random_forest']:.3f})")
    for r in sorted(results, key=lambda r: -r["test_auc"]):
        star = "  <-- FULL" if r["views"] == "x+tree+deep" else ""
        print(f"  {r['model']:22s} {r['test_auc']:.4f}   acc={r['test_acc']:.3f}{star}")
    full = next(r for r in results if r["views"] == "x+tree+deep")
    raw = next(r for r in results if r["views"] == "x")
    print(f"\n  FULL vs raw-x baseline: {full['test_auc']:.4f} vs {raw['test_auc']:.4f} "
          f"({full['test_auc'] - raw['test_auc']:+.4f})")
    print(f"  FULL vs tree ceiling:   {full['test_auc']:.4f} vs {tree_ceiling:.4f} "
          f"({full['test_auc'] - tree_ceiling:+.4f})")

    # -------- persist --------
    summary = dict(dataset=ds.name, task=args.task, seed=args.seed, fusion=args.fusion,
                   tree_ceiling=tree_ceiling, ceiling=ceil, tree_encoding_width=E,
                   results=results)
    with open(os.path.join(args.out, f"fusion_{ds.name}.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # -------- figure --------
    names = ["tree\nceiling"] + [r["model"].replace(" ", "\n") for r in results]
    vals = [tree_ceiling] + [r["test_auc"] for r in results]
    cols = ["#2ca02c"] + ["#d62728" if r["views"] == "x+tree+deep" else "#1f77b4"
                          for r in results]
    fig, axp = plt.subplots(figsize=(max(7, 1.6 * len(names)), 5))
    axp.bar(range(len(names)), vals, color=cols, edgecolor="white")
    axp.axhline(tree_ceiling, ls="--", c="green", lw=1, alpha=0.6)
    axp.set_xticks(range(len(names))); axp.set_xticklabels(names, fontsize=9)
    axp.set_ylim(0.5, max(0.75, tree_ceiling + 0.02)); axp.set_ylabel("test AUC")
    axp.set_title(f"Three-view fusion — {ds.name} (fusion={args.fusion})")
    for i, v in enumerate(vals):
        axp.text(i, v + 0.003, f"{v:.3f}", ha="center", fontsize=8)
    fig.tight_layout()
    png = os.path.join(args.out, f"fusion_{ds.name}.png")
    fig.savefig(png, dpi=140, bbox_inches="tight")
    print(f"\n[fusion] figure  -> {png}")
    print(f"[fusion] summary -> {args.out}/fusion_{ds.name}.json")


if __name__ == "__main__":
    main()
