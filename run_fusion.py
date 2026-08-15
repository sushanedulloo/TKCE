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
import pandas as pd
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


def split_direction_encoding(rf, X, tau=0.0):
    """For every internal split node across all trees: does the sample go LEFT?

    tau == 0 : hard bits, 1[x_feat <= threshold]  (DeepTLF-style encoding)
    tau  > 0 : SOFT bits, sigmoid((threshold - x_feat)/tau) — a temperature-
               controlled relaxation of the fitted forest's routing. Features
               are standardized upstream, so tau ~ 0.1-1.0 is a sensible range.
               Soft bits carry "how close to the boundary" information and are
               far less fingerprint-like than hard bits (anti-memorization).
    """
    X = np.ascontiguousarray(X)
    cols = []
    for est in rf.estimators_:
        t = est.tree_
        internal = np.where(t.feature >= 0)[0]          # split nodes only
        feats = t.feature[internal]
        thr = t.threshold[internal].astype(np.float32)
        margin = thr - X[:, feats]                      # >0 means "goes left"
        if tau > 0:
            bits = 1.0 / (1.0 + np.exp(-margin / tau))
        else:
            bits = (margin >= 0)
        cols.append(bits.astype(np.float32))
    return np.concatenate(cols, axis=1) if cols else np.zeros((len(X), 0), np.float32)


def oob_honest_encoding(rf, enc_train, n_train):
    """OOB-honest ("out-of-bag dropout") training encoding.

    The forest was fit ON the training rows' labels, so a training row's bits
    from trees whose bootstrap sample CONTAINED that row are contaminated: the
    tree's splits were partly carved to classify that very row. Test rows have
    no such trees — a distribution shift that lets the downstream net memorize.

    Fix: for each training row keep only the bits from trees where the row was
    OUT-OF-BAG (~37% of trees, honest by construction), zero the rest, and
    rescale by T/|OOB| (inverted-dropout convention) so expected magnitudes
    match the all-trees encoding used for val/test. Costs nothing: bootstrap
    masks are a free by-product of the RF.
    """
    T = len(rf.estimators_)
    inbag = np.zeros((n_train, T), dtype=bool)
    for t, samp in enumerate(rf.estimators_samples_):
        inbag[np.unique(samp), t] = True
    oob = ~inbag
    n_oob = oob.sum(axis=1).clip(min=1)
    counts = [int((est.tree_.feature >= 0).sum()) for est in rf.estimators_]
    colmask = np.repeat(oob, counts, axis=1).astype(np.float32)
    scale = (T / n_oob).astype(np.float32)[:, None]
    return enc_train * colmask * scale, float(n_oob.mean())


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
def evaluate(model, X, T, y, device, bs=8192, return_proba=False):
    model.eval()
    ps, tot = [], 0.0
    for s in range(0, len(X), bs):
        xb = torch.from_numpy(X[s:s + bs]).to(device)
        tb = torch.from_numpy(T[s:s + bs]).to(device)
        yb = torch.from_numpy(y[s:s + bs]).to(device)
        out = model(xb, tb)
        tot += F.cross_entropy(out, yb, reduction="sum").item()
        ps.append(F.softmax(out, dim=1).cpu().numpy())
    proba = np.concatenate(ps, axis=0)
    m = clf_metrics(y, proba)
    m["loss"] = tot / len(X)
    return (m, proba) if return_proba else m


def train_one(views, ds, enc, n_classes, args, device, label, member=0):
    seed = args.seed + 1000 * member
    torch.manual_seed(seed); np.random.seed(seed)
    model = FusionModel(ds.n_features, enc["train"].shape[1], n_classes,
                        views, args).to(device)
    n_par = sum(p.numel() for p in model.parameters())
    print(f"\n[{label}]{f' member {member+1}/{args.ensemble}' if args.ensemble > 1 else ''} "
          f"views={views}  concat_dim={model.cat_dim}  params={n_par/1e6:.2f}M",
          flush=True)

    loader = DataLoader(
        TensorDataset(torch.from_numpy(ds.X_train).to(device),
                      torch.from_numpy(enc["train"]).to(device),
                      torch.from_numpy(ds.y_train).to(device)),
        batch_size=args.batch_size, shuffle=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay)

    best_auc, best_state, best_ep, hist = -1.0, None, 0, []
    for epoch in range(1, args.epochs + 1):
        model.train()
        for xb, tb, yb in loader:
            opt.zero_grad()
            loss = F.cross_entropy(model(xb, tb), yb)
            if args.l1 > 0:                                   # L1 on weight matrices
                loss = loss + args.l1 * sum(p.abs().sum()
                                            for p in model.parameters() if p.ndim >= 2)
            loss.backward(); opt.step()
        tr = evaluate(model, ds.X_train, enc["train"], ds.y_train, device)
        va = evaluate(model, ds.X_val, enc["val"], ds.y_val, device)
        hist.append(dict(model=label, epoch=epoch,
                         train_loss=tr["loss"], val_loss=va["loss"],
                         train_auc=tr["auc"], val_auc=va["auc"],
                         train_acc=tr["accuracy"], val_acc=va["accuracy"]))
        if va["auc"] > best_auc:
            best_auc, best_ep = va["auc"], epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if epoch % max(1, args.epochs // 8) == 0:
            print(f"    epoch {epoch:4d}/{args.epochs}  train_auc={tr['auc']:.4f}  "
                  f"val_auc={va['auc']:.4f}  (best {best_auc:.4f} @ {best_ep})", flush=True)

    model.load_state_dict(best_state)
    te, te_proba = evaluate(model, ds.X_test, enc["test"], ds.y_test, device,
                            return_proba=True)
    print(f"  -> {label:22s} TEST auc={te['auc']:.4f} acc={te['accuracy']:.4f} "
          f"(best val {best_auc:.4f} @ epoch {best_ep})", flush=True)
    res = dict(model=label, views="+".join(views), member=member,
               test_auc=te["auc"], test_acc=te["accuracy"],
               best_val_auc=best_auc, best_epoch=best_ep,
               concat_dim=model.cat_dim, params_M=round(n_par / 1e6, 2))
    return res, hist, te_proba


def run_config(views, ds, enc, n_classes, args, device, label):
    """Train `--ensemble` members (different seeds) and average their test
    probabilities — a plain deep ensemble, the best-evidenced cheap
    regularizer for tabular MLP-family models (cf. TabM)."""
    members, hists, probs = [], [], []
    for m in range(args.ensemble):
        res, hist, p = train_one(views, ds, enc, n_classes, args, device,
                                 label, member=m)
        members.append(res); hists.extend(hist); probs.append(p)
    if args.ensemble == 1:
        return members[0], hists
    em = clf_metrics(ds.y_test, np.mean(probs, axis=0))
    res = dict(model=label, views="+".join(views),
               test_auc=em["auc"], test_acc=em["accuracy"],
               best_val_auc=float(np.mean([r["best_val_auc"] for r in members])),
               best_epoch=int(np.mean([r["best_epoch"] for r in members])),
               concat_dim=members[0]["concat_dim"],
               params_M=members[0]["params_M"], ensemble=args.ensemble,
               member_test_aucs=[round(r["test_auc"], 4) for r in members])
    member_str = ", ".join(f"{r['test_auc']:.4f}" for r in members)
    print(f"  => {label:22s} ENSEMBLE({args.ensemble}) TEST auc={em['auc']:.4f} "
          f"acc={em['accuracy']:.4f}  (members: {member_str})", flush=True)
    return res, hists


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", type=int, default=361070)          # eye_movements
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-rows", type=int, default=16000)
    # data hygiene (leak repair)
    ap.add_argument("--drop-cols", default="",
                    help="comma-separated feature names to REMOVE (leaky ID "
                         "columns; eye_movements: lineNo,assgNo,titleNo,wordNo)")
    ap.add_argument("--group-by", default="",
                    help="feature name to GROUP the train/val/test split by "
                         "(no group shared across splits; eye_movements: assgNo)")
    # encoding options
    ap.add_argument("--tau", type=float, default=0.0,
                    help="soft-encoding temperature; 0 = hard bits, try 0.1-1.0 "
                         "(features are standardized)")
    ap.add_argument("--encoding", default="infold", choices=["infold", "oob"],
                    help="infold = naive (forest saw the rows it encodes; leaks "
                         "labels into train bits). oob = OOB-honest: train rows "
                         "keep only bits from trees they were out-of-bag for")
    # ensembling + run selection
    ap.add_argument("--ensemble", type=int, default=1,
                    help="train k members per config (different seeds) and "
                         "average predictions (deep ensemble)")
    ap.add_argument("--views", default="",
                    help="semicolon-separated view configs to run, e.g. "
                         "'x;x+tree'. Overrides --ablation. Empty = default set")
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
    ap.add_argument("--l1", type=float, default=0.0,
                    help="L1 weight penalty (sparsity). Targets the big tree-encoding "
                         "projection to prune useless split bits. Try 1e-5..1e-4.")
    # misc
    ap.add_argument("--ablation", action="store_true",
                    help="also run x+tree and x+deep to isolate each view")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", default="results/fusion")
    args = ap.parse_args()

    device = resolve_device(args.device)
    os.makedirs(args.out, exist_ok=True)
    drop_cols = [c.strip() for c in args.drop_cols.split(",") if c.strip()] or None
    group_col = args.group_by.strip() or None
    ds = load_task(args.task, seed=args.seed, max_rows=args.max_rows,
                   drop_cols=drop_cols, group_col=group_col)
    if ds.task_type != "classification":
        raise SystemExit(f"{ds.name} is not classification.")
    C = ds.n_classes
    print(f"\n=== FUSION: {ds.name} (task {args.task}) | {ds.n_features} feats, "
          f"{C} classes | fusion={args.fusion} | device={device} ===", flush=True)
    print(f"[data] split={ds.meta['split']}"
          f"{f' by {group_col}' if group_col else ''}"
          f"{f' | dropped: {drop_cols}' if drop_cols else ''} | "
          f"train={len(ds.y_train)} val={len(ds.y_val)} test={len(ds.y_test)}",
          flush=True)
    print(f"[reg] lr={args.lr:g} batch={args.batch_size} dropout={args.dropout} "
          f"weight_decay={args.weight_decay:g} l1={args.l1:g} "
          f"encoding={args.encoding} tau={args.tau:g} ensemble={args.ensemble}",
          flush=True)

    # -------- view 2: RF split-direction encoding --------
    print(f"[view2] fitting encoding RF ({args.rf_trees} trees, depth {args.rf_depth}) ...",
          flush=True)
    rf = fit_encoder_rf(ds.X_train, ds.y_train, args)
    enc = {"train": split_direction_encoding(rf, ds.X_train, args.tau),
           "val":   split_direction_encoding(rf, ds.X_val, args.tau),
           "test":  split_direction_encoding(rf, ds.X_test, args.tau)}
    E = enc["train"].shape[1]
    print(f"[view2] tree-encoding width = {E} split bits "
          f"({'hard' if args.tau == 0 else f'soft tau={args.tau:g}'}, "
          f"~{E * enc['train'].shape[0] * 4 / 1e6:.0f} MB train)", flush=True)
    if args.encoding == "oob":
        enc["train"], mean_oob = oob_honest_encoding(rf, enc["train"],
                                                     len(ds.y_train))
        print(f"[view2] OOB-honest training encoding: each train row keeps bits "
              f"from {mean_oob:.1f}/{args.rf_trees} trees on average "
              f"(val/test keep all trees)", flush=True)

    # -------- tree ceiling (reference) --------
    print("[ceiling] fitting tree baselines ...", flush=True)
    ceil = {}
    for name in ("xgboost", "lightgbm", "random_forest"):
        te, _, _ = fit_tree_baseline(name, ds, {"seed": args.seed})
        ceil[name] = te["auc"]
        print(f"  {name:14s} AUC={te['auc']:.4f}", flush=True)
    tree_ceiling = max(ceil.values())

    # -------- neural runs --------
    if args.views:
        run_views = [v.strip().split("+") for v in args.views.split(";") if v.strip()]
    elif args.ablation:
        run_views = [["x"], ["x", "tree"], ["x", "deep"], ["x", "tree", "deep"]]
    else:
        run_views = [["x"], ["x", "tree", "deep"]]
    labels = {"x": "raw (x only)", "x+tree": "x + tree", "x+deep": "x + deep",
              "x+tree+deep": "FULL (x+tree+deep)"}
    results, hists = [], []
    for v in run_views:
        lab = labels.get("+".join(v), "+".join(v))
        res, h = run_config(v, ds, enc, C, args, device, lab)
        results.append(res); hists.extend(h)
    hdf = pd.DataFrame(hists)
    hdf.to_csv(os.path.join(args.out, f"fusion_{ds.name}_epochs.csv"), index=False)

    # -------- summary --------
    print("\n" + "=" * 64)
    print(f"RESULTS — {ds.name} | fusion={args.fusion} | epochs={args.epochs}")
    print("=" * 64)
    print(f"  {'tree ceiling':22s} {tree_ceiling:.4f}   "
          f"(xgb {ceil['xgboost']:.3f} / lgb {ceil['lightgbm']:.3f} / rf {ceil['random_forest']:.3f})")
    best = max(results, key=lambda r: r["test_auc"])
    for r in sorted(results, key=lambda r: -r["test_auc"]):
        star = "  <-- best neural" if r is best else ""
        print(f"  {r['model']:22s} {r['test_auc']:.4f}   acc={r['test_acc']:.3f}{star}")
    raw = next((r for r in results if r["views"] == "x"), None)
    if raw is not None and raw is not best:
        print(f"\n  best neural vs raw-x:   {best['test_auc']:.4f} vs "
              f"{raw['test_auc']:.4f} ({best['test_auc'] - raw['test_auc']:+.4f})")
    print(f"  best neural vs ceiling: {best['test_auc']:.4f} vs {tree_ceiling:.4f} "
          f"({best['test_auc'] - tree_ceiling:+.4f})")

    # -------- persist --------
    summary = dict(dataset=ds.name, task=args.task, seed=args.seed, fusion=args.fusion,
                   lr=args.lr, batch_size=args.batch_size, dropout=args.dropout,
                   weight_decay=args.weight_decay, l1=args.l1,
                   encoding=args.encoding, tau=args.tau, ensemble=args.ensemble,
                   drop_cols=drop_cols or [], group_by=group_col,
                   split=ds.meta["split"],
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

    # ---- training-curve figure: loss + AUC over epochs, all models overlaid ----
    model_order = [r["model"] for r in results]
    cmap = plt.get_cmap("tab10")
    mcol = {m: cmap(i % 10) for i, m in enumerate(model_order)}
    figc, axc = plt.subplots(2, 2, figsize=(15, 10))

    def curve(ax, col, title, ylab, hline=None):
        for m in model_order:
            g = hdf[hdf.model == m].groupby("epoch")[col].mean()  # mean over members
            ax.plot(g.index, g.values, color=mcol[m], lw=1.4, label=m)
        if hline is not None:
            ax.axhline(hline, ls="--", c="green", lw=1, alpha=0.6, label="tree ceiling")
        ax.set_title(title); ax.set_xlabel("epoch"); ax.set_ylabel(ylab)
        ax.grid(alpha=0.3); ax.legend(fontsize=7)

    curve(axc[0, 0], "val_auc", "Validation AUC vs epoch", "AUC", tree_ceiling)
    curve(axc[0, 1], "train_auc", "Train AUC vs epoch (overfitting gauge)", "AUC", tree_ceiling)
    curve(axc[1, 0], "train_loss", "Train loss vs epoch", "cross-entropy")
    curve(axc[1, 1], "val_loss", "Validation loss vs epoch (up = overfitting)", "cross-entropy")
    figc.suptitle(f"Training curves — {ds.name} (fusion={args.fusion})",
                  fontsize=13, fontweight="bold")
    figc.tight_layout(rect=[0, 0, 1, 0.97])
    cpng = os.path.join(args.out, f"fusion_{ds.name}_curves.png")
    figc.savefig(cpng, dpi=140, bbox_inches="tight")

    print(f"\n[fusion] bar chart    -> {png}")
    print(f"[fusion] curves       -> {cpng}")
    print(f"[fusion] epoch curves -> {args.out}/fusion_{ds.name}_epochs.csv")
    print(f"[fusion] summary      -> {args.out}/fusion_{ds.name}.json")


if __name__ == "__main__":
    main()
