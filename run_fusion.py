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
    return encode_margins(split_margins(rf, X), tau)


def split_margins(rf, X):
    """margin = threshold - x_feat for every internal split node (>0 = goes left).
    The margin is the raw material for both hard and soft bits, and for the
    train-time Gaussian-noise augmentation (--enc-noise)."""
    X = np.ascontiguousarray(X)
    cols = []
    for est in rf.estimators_:
        t = est.tree_
        internal = np.where(t.feature >= 0)[0]          # split nodes only
        feats = t.feature[internal]
        thr = t.threshold[internal].astype(np.float32)
        cols.append((thr - X[:, feats]).astype(np.float32))
    return np.concatenate(cols, axis=1) if cols else np.zeros((len(X), 0), np.float32)


def encode_margins(M, tau=0.0):
    if tau > 0:
        return (1.0 / (1.0 + np.exp(-M / tau))).astype(np.float32)
    return (M >= 0).astype(np.float32)


def node_depths_and_samples(rf):
    """Depth + training-sample count for every internal split node, in the SAME
    column order that split_margins uses. Deep nodes are fit on few samples —
    they are the rote-memorization part of the encoding."""
    depths, samples = [], []
    for est in rf.estimators_:
        t = est.tree_
        d = np.zeros(t.node_count, dtype=np.int32)
        stack = [(0, 0)]
        while stack:
            i, dep = stack.pop()
            d[i] = dep
            if t.children_left[i] != -1:
                stack.append((t.children_left[i], dep + 1))
                stack.append((t.children_right[i], dep + 1))
        internal = np.where(t.feature >= 0)[0]
        depths.append(d[internal])
        samples.append(t.n_node_samples[internal].astype(np.int64))
    return np.concatenate(depths), np.concatenate(samples)


def select_bits(rf, frac=None, layers=None, select="deepest", seed=0):
    """Boolean mask over the encoding columns.

    WHICH bits: `layers` (explicit depth set, e.g. [5] or [0,1,2]) or `frac`
    (fraction of all bits).  HOW chosen:
      deepest    -> deepest first, ties by fewest training samples (most memorized)
      shallowest -> the mirror image (the reverse arm)
      random     -> a random subset of the SAME SIZE — the matched control that
                    separates "depth matters" from "the encoding is redundant"
    """
    depths, samples = node_depths_and_samples(rf)
    n = len(depths)
    k = int(np.isin(depths, layers).sum()) if layers is not None \
        else max(1, int(round(frac * n)))
    if select == "deepest":
        order = np.lexsort((samples, -depths))
    elif select == "shallowest":
        order = np.lexsort((-samples, depths))
    elif select == "random":
        order = np.random.default_rng(seed).permutation(n)
    else:
        raise ValueError(select)
    mask = np.zeros(n, dtype=bool)
    mask[order[:k]] = True
    return mask, depths, samples


def deepest_bits_mask(rf, frac):
    return select_bits(rf, frac=frac, select="deepest")


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
    ms, mean_oob = oob_mask_matrix(rf, n_train)
    return enc_train * ms, mean_oob


def oob_mask_matrix(rf, n_train):
    """The (n_train, n_bits) multiplier that implements OOB honesty: zero for
    in-bag trees' bits, T/|OOB| (inverted-dropout rescale) for kept ones."""
    T = len(rf.estimators_)
    inbag = np.zeros((n_train, T), dtype=bool)
    for t, samp in enumerate(rf.estimators_samples_):
        inbag[np.unique(samp), t] = True
    oob = ~inbag
    n_oob = oob.sum(axis=1).clip(min=1)
    counts = [int((est.tree_.feature >= 0).sum()) for est in rf.estimators_]
    colmask = np.repeat(oob, counts, axis=1).astype(np.float32)
    scale = (T / n_oob).astype(np.float32)[:, None]
    return colmask * scale, float(n_oob.mean())


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


def train_one(views, ds, enc, n_classes, args, device, label, member=0, noise=None):
    seed = args.seed + 1000 * member
    torch.manual_seed(seed); np.random.seed(seed)
    model = FusionModel(ds.n_features, enc["train"].shape[1], n_classes,
                        views, args).to(device)
    n_par = sum(p.numel() for p in model.parameters())
    use_noise = noise is not None and "tree" in views
    print(f"\n[{label}]{f' member {member+1}/{args.ensemble}' if args.ensemble > 1 else ''} "
          f"views={views}  concat_dim={model.cat_dim}  params={n_par/1e6:.2f}M"
          f"{('  enc-noise sigma=%g (fresh each batch)' % args.enc_noise) if use_noise and noise['kind'] == 'margin_noise' else ''}"
          f"{('  bit-flip on %d bits, mean p=%.3f (fresh each batch)' % (int((noise['pvec'] > 0).sum()), float(noise['pvec'].mean()))) if use_noise and noise['kind'] == 'bit_flip' else ''}",
          flush=True)

    pvec_t = None
    if use_noise:
        # loader carries the RAW material (margins or hard bits); the noisy /
        # flipped bits are made fresh every batch, then the OOB mask is applied.
        # eval always uses the clean encoding in `enc` — augmentation is train-only.
        ms = noise.get("maskscale")
        raw = noise["margins"] if noise["kind"] == "margin_noise" else noise["bits"]  # bit_flip -> hard bits
        if noise["kind"] == "bit_flip":
            pvec_t = torch.from_numpy(noise["pvec"]).to(device).unsqueeze(0)
        tensors = [torch.from_numpy(ds.X_train).to(device),
                   torch.from_numpy(raw).to(device)]
        if ms is not None:
            tensors.append(torch.from_numpy(ms).to(device))
        tensors.append(torch.from_numpy(ds.y_train).to(device))
        loader = DataLoader(TensorDataset(*tensors),
                            batch_size=args.batch_size, shuffle=True)
    else:
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
        for batch in loader:
            if use_noise:
                if len(batch) == 4:
                    xb, mb, msb, yb = batch
                else:
                    (xb, mb, yb), msb = batch, None
                if noise["kind"] == "margin_noise":
                    tb = torch.sigmoid(
                        (mb + args.enc_noise * torch.randn_like(mb)) / args.tau)
                else:                                  # bit_flip: mb = hard bits
                    flips = torch.rand_like(mb) < pvec_t
                    tb = torch.where(flips, 1.0 - mb, mb)
                if msb is not None:
                    tb = tb * msb
            else:
                xb, tb, yb = batch
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


def run_config(views, ds, enc, n_classes, args, device, label, noise=None):
    """Train `--ensemble` members (different seeds) and average their test
    probabilities — a plain deep ensemble, the best-evidenced cheap
    regularizer for tabular MLP-family models (cf. TabM)."""
    members, hists, probs = [], [], []
    for m in range(args.ensemble):
        res, hist, p = train_one(views, ds, enc, n_classes, args, device,
                                 label, member=m, noise=noise)
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
    ap.add_argument("--deep-frac", type=float, default=0.1,
                    help="fraction of DEEPEST split bits targeted by --deep-flip-p "
                         "/ --deep-delete (professor: 'the last 10%%')")
    ap.add_argument("--deep-flip-p", type=float, default=0.0,
                    help="train-time flip probability (0<->1) for the deepest bits, "
                         "fresh each batch; bits stay binary; eval uses clean bits. "
                         "p=0.5 = replace those bits with pure coin flips")
    ap.add_argument("--deep-delete", action="store_true",
                    help="zero out the deepest bits entirely (train AND eval) — the "
                         "diagnostic arm: if deep bits are pure memorization, "
                         "deleting them should barely hurt test")
    ap.add_argument("--deep-select", choices=["deepest", "random", "shallowest"],
                    default="deepest",
                    help="how the targeted bits are chosen: deepest (default), "
                         "shallowest (the reverse arm), or random of the same "
                         "size (the matched control)")
    ap.add_argument("--deep-layers", type=str, default=None,
                    help="target whole depth layers instead of a fraction, e.g. "
                         "'5' (bottom layer), '4,5', '0,1,2'; with --deep-select "
                         "random -> a random subset of the same SIZE")
    ap.add_argument("--flip-ramp", type=float, default=0.0,
                    help="depth-ramped bit flips on ALL bits: p(depth) = P * "
                         "depth/max_depth (roots never flip, bottom layer gets P); "
                         "train-only, fresh each batch, bits stay binary")
    ap.add_argument("--flip-uniform", type=float, default=0.0,
                    help="uniform bit flips on ALL bits with probability P — the "
                         "matched-budget control for --flip-ramp")
    ap.add_argument("--enc-noise", type=float, default=0.0,
                    help="train-time Gaussian noise on the split MARGINS, squashed "
                         "by the sigmoid: bits = sigmoid((margin + sigma*eps)/tau). "
                         "Fresh noise every batch; eval uses the clean encoding. "
                         "Requires --tau > 0. Try 0.1-0.4 (features are standardized)")
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
          f"encoding={args.encoding} tau={args.tau:g} enc_noise={args.enc_noise:g} "
          f"ensemble={args.ensemble}",
          flush=True)

    # -------- view 2: RF split-direction encoding --------
    print(f"[view2] fitting encoding RF ({args.rf_trees} trees, depth {args.rf_depth}) ...",
          flush=True)
    if args.enc_noise > 0 and args.tau <= 0:
        raise SystemExit("--enc-noise needs --tau > 0 (the sigmoid that squashes "
                         "the noisy margins back into (0,1)); try --tau 0.1")
    n_flip = sum([args.deep_flip_p > 0, args.flip_ramp > 0, args.flip_uniform > 0])
    if n_flip > 1:
        raise SystemExit("pick ONE of --deep-flip-p / --flip-ramp / --flip-uniform")
    if n_flip and args.deep_delete:
        raise SystemExit("bit flips cannot be combined with --deep-delete")
    if n_flip and args.enc_noise > 0:
        raise SystemExit("pick ONE of bit flips / --enc-noise")
    if n_flip and args.tau != 0:
        raise SystemExit("bit flips work on HARD bits; drop --tau "
                         "(flipping soft bits would re-introduce the softness confound)")
    rf = fit_encoder_rf(ds.X_train, ds.y_train, args)
    M_train = split_margins(rf, ds.X_train)
    enc = {"train": encode_margins(M_train, args.tau),
           "val":   split_direction_encoding(rf, ds.X_val, args.tau),
           "test":  split_direction_encoding(rf, ds.X_test, args.tau)}
    E = enc["train"].shape[1]
    print(f"[view2] tree-encoding width = {E} split bits "
          f"({'hard' if args.tau == 0 else f'soft tau={args.tau:g}'}, "
          f"~{E * enc['train'].shape[0] * 4 / 1e6:.0f} MB train)", flush=True)
    noise_pack = None
    maskscale = None
    if args.encoding == "oob":
        maskscale, mean_oob = oob_mask_matrix(rf, len(ds.y_train))
        enc["train"] = enc["train"] * maskscale
        print(f"[view2] OOB-honest training encoding: each train row keeps bits "
              f"from {mean_oob:.1f}/{args.rf_trees} trees on average "
              f"(val/test keep all trees)", flush=True)
    if args.enc_noise > 0:
        noise_pack = {"kind": "margin_noise", "margins": M_train, "maskscale": maskscale}
        print(f"[view2] train-time margin noise: sigma={args.enc_noise:g}, "
              f"resampled every batch; evaluation uses the clean encoding",
              flush=True)
    deep_layers = ([int(v) for v in args.deep_layers.split(",")]
                   if args.deep_layers else None)
    deep_info = {}
    if args.deep_flip_p > 0 or args.deep_delete:
        dmask, depths, samp = select_bits(
            rf, frac=None if deep_layers else args.deep_frac, layers=deep_layers,
            select=args.deep_select, seed=args.seed)
        tgt, rest = samp[dmask], samp[~dmask]
        what = f"layers {deep_layers}" if deep_layers else f"{args.deep_frac:.0%}"
        print(f"[deep] targeting {args.deep_select} {what}: "
              f"{int(dmask.sum())}/{len(dmask)} bits ({dmask.mean():.1%}), depths "
              f"{int(depths[dmask].min())}-{int(depths[dmask].max())} "
              f"(vs {int(depths.min())}-{int(depths.max())} overall)", flush=True)
        print(f"[deep] median training samples per targeted node: {int(np.median(tgt))} "
              f"vs {int(np.median(rest))} for the rest "
              f"<- few samples = rote memorization", flush=True)
        deep_info = dict(deep_n_bits=int(dmask.sum()), n_bits_total=int(len(dmask)),
                         deep_frac_actual=float(dmask.mean()),
                         deep_depth_min=int(depths[dmask].min()),
                         deep_depth_max=int(depths[dmask].max()),
                         deep_median_samples=int(np.median(tgt)),
                         rest_median_samples=int(np.median(rest)))
        if args.deep_delete:
            for k in enc:
                enc[k][:, dmask] = 0.0
            print(f"[deep] DELETE mode: those bits are zeroed everywhere "
                  f"(train + val + test)", flush=True)
        else:
            pvec = np.where(dmask, args.deep_flip_p, 0.0).astype(np.float32)
            noise_pack = {"kind": "bit_flip", "bits": encode_margins(M_train, 0.0),
                          "maskscale": maskscale, "pvec": pvec}
            print(f"[deep] FLIP mode: each targeted bit flips 0<->1 with "
                  f"p={args.deep_flip_p:g}, fresh every batch (train only; "
                  f"p=0.5 = pure coin flips)", flush=True)
    if args.flip_ramp > 0 or args.flip_uniform > 0:
        depths, _samp = node_depths_and_samples(rf)
        if args.flip_ramp > 0:
            pvec = (args.flip_ramp * depths / depths.max()).astype(np.float32)
            mode = f"RAMP p(depth) = {args.flip_ramp:g} * depth/{int(depths.max())}"
        else:
            pvec = np.full(len(depths), args.flip_uniform, dtype=np.float32)
            mode = f"UNIFORM p = {args.flip_uniform:g} on every bit"
        per_depth = {int(d): (float(pvec[depths == d][0]), int((depths == d).sum()))
                     for d in np.unique(depths)}
        print(f"[flip] {mode}; train-only, fresh each batch, bits stay binary", flush=True)
        print("[flip] per depth (p, #bits): " + ", ".join(
            f"d{d}: ({pp:.3f}, {nn})" for d, (pp, nn) in per_depth.items()), flush=True)
        print(f"[flip] average flip rate over all {len(pvec)} bits = {pvec.mean():.4f} "
              f"<- the matched uniform control uses this number", flush=True)
        deep_info = dict(n_bits_total=int(len(pvec)), flip_mean_p=float(pvec.mean()),
                         flip_per_depth={str(d): pp for d, (pp, _) in per_depth.items()})
        noise_pack = {"kind": "bit_flip", "bits": encode_margins(M_train, 0.0),
                      "maskscale": maskscale, "pvec": pvec}

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
        res, h = run_config(v, ds, enc, C, args, device, lab, noise=noise_pack)
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
                   encoding=args.encoding, tau=args.tau, enc_noise=args.enc_noise,
                   deep_frac=args.deep_frac, deep_flip_p=args.deep_flip_p,
                   deep_delete=bool(args.deep_delete),
                   deep_select=args.deep_select, deep_layers=deep_layers,
                   flip_ramp=args.flip_ramp, flip_uniform=args.flip_uniform,
                   **deep_info,
                   ensemble=args.ensemble,
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
