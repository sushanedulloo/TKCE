"""Deep heads on a regularised tree encoding: does extra capacity buy anything?

The feature side is settled: deleting the deepest tree layers stops the model
memorising, on every dataset where it was memorising. This script asks the next
question. Keeping the features fixed, replace the linear head with a deep
network and see whether its extra capacity turns into better predictions, or
just into a new way to overfit -- and whether the network needs regularising of
its own on top of the regularised features.

    same setup, only the head changes

Heads
    linear      scikit-learn logistic regression, the reference. The penalty C
                is swept and chosen on validation, so a deep head cannot win
                merely by being better tuned.
    mlp         a plain multi-layer network on the bits
    tabresnet   the residual network used earlier in this project
    treetf      a transformer whose tokens are TREES, not bits (see below)

Why tree tokens. A transformer lets every input attend to every other, at a
cost that grows with the square of the token count; 1,454 bits would be far too
slow. But the encoding already has structure: each bit belongs to exactly one
tree. Bundling a tree's bits into one token gives 100 tokens, one per tree, and
the attention then learns how trees relate to one another. The projection from
a tree's bits to its token is shared across trees, because trees are
interchangeable, and a learned per-tree embedding is added so the model can
still tell them apart.

Everything is reported on train, validation and test: ranking score, accuracy
and log loss, plus the train-minus-validation gap on the ranking score. The
training score is computed on CLEAN bits.

Resuming. The fitted forest, the bits and the tree baselines are cached to disk
(point --cache-dir at Google Drive) so a second session does not recompute
them. Training checkpoints every --ckpt-every epochs, and a rerun continues
from the last checkpoint instead of starting over.

    python run_deep_head.py --task 361065 --drop-layers 4,5 --head linear
    python run_deep_head.py --task 361065 --drop-layers 4,5 --head treetf --size medium
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import time
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from torch.utils.data import DataLoader, TensorDataset

from run_fusion import (encode_margins, node_depths_and_samples, oob_mask_matrix,
                        split_direction_encoding, split_margins)
from tkce.baselines import fit_tree_baseline
from tkce.data import load_task
from tkce.models import MLPHead, TabResNet

warnings.filterwarnings("ignore", message="X does not have valid feature names")
# scikit-learn 1.8 deprecates n_jobs on LogisticRegression; nothing here sets it,
# but a dependency may, and the warning would repeat through every run log.
warnings.filterwarnings("ignore", message=".*n_jobs.*has no effect.*")
# norm_first disables PyTorch's nested-tensor fast path; that is expected and
# the warning would repeat in every transformer run log.
warnings.filterwarnings("ignore", message=".*enable_nested_tensor.*")


# --------------------------------------------------------------------------- #
# the transformer over BIT tokens -- every bit attends to every other bit
# --------------------------------------------------------------------------- #
class _AttnBlock(nn.Module):
    """One pre-norm transformer block using PyTorch's memory-efficient attention.

    Written by hand rather than with nn.TransformerEncoderLayer so the call to
    scaled_dot_product_attention is explicit. That matters here: the ordinary
    attention implementation materialises a (batch, heads, n, n) score matrix,
    which for 1,455 bit tokens at batch 256 is about 17 GB and will not fit on
    a T4. The fused kernel never builds that matrix, so memory grows linearly
    in the number of tokens instead of quadratically, and only the arithmetic
    stays quadratic.
    """

    def __init__(self, d, n_heads, dropout):
        super().__init__()
        self.h, self.dh, self.p = n_heads, d // n_heads, dropout
        self.n1 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.n2 = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(),
                                nn.Dropout(dropout), nn.Linear(2 * d, d))
        self.drop = nn.Dropout(dropout)

    def forward(self, x):                                   # (B, N, D)
        B, N, D = x.shape
        q, k, v = (self.qkv(self.n1(x))
                   .view(B, N, 3, self.h, self.dh).permute(2, 0, 3, 1, 4))
        a = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.p if self.training else 0.0)
        x = x + self.drop(self.proj(a.transpose(1, 2).reshape(B, N, D)))
        return x + self.ff(self.n2(x))


class BitTokenTransformer(nn.Module):
    """Every bit is its own token; attention runs over all of them.

    Each bit j gets its own learned embedding direction, so token_j is
    value_j * w_j + b_j. That is the feature-tokenizer idea: a bit that is on
    contributes w_j + b_j, a bit that is off contributes b_j, and the model
    learns what each individual split means. A classification token is
    prepended and its output feeds the prediction.
    """

    def __init__(self, n_bits, out_dim, d_token=64, n_blocks=3, n_heads=8,
                 dropout=0.1):
        super().__init__()
        self.w = nn.Parameter(torch.randn(n_bits, d_token) * 0.02)
        self.b = nn.Parameter(torch.zeros(n_bits, d_token))
        self.cls = nn.Parameter(torch.randn(1, 1, d_token) * 0.02)
        self.blocks = nn.ModuleList(
            [_AttnBlock(d_token, n_heads, dropout) for _ in range(n_blocks)])
        self.norm = nn.LayerNorm(d_token)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(d_token, out_dim)

    def forward(self, x):                                   # (B, n_bits)
        tok = x.unsqueeze(-1) * self.w + self.b             # (B, n_bits, d)
        tok = torch.cat([self.cls.expand(x.shape[0], -1, -1), tok], dim=1)
        for blk in self.blocks:
            tok = blk(tok)
        return self.head(self.drop(self.norm(tok[:, 0])))


# --------------------------------------------------------------------------- #
# the transformer over TREE tokens (kept as a cheaper comparison arm)
# --------------------------------------------------------------------------- #
def tree_token_index(rf, kept):
    """Map the surviving bits onto a (n_trees, max_bits_per_tree) index grid.

    Returns `idx` (int64) holding, for each tree, the column positions of its
    surviving bits inside the kept-bit vector, padded with 0, and `mask` (bool)
    marking which of those entries are real. Padding is masked, never read as
    a value, so trees with fewer bits are handled exactly.
    """
    counts = [int((est.tree_.feature >= 0).sum()) for est in rf.estimators_]
    tree_of_bit = np.repeat(np.arange(len(counts)), counts)[kept]
    n_trees = len(counts)
    per_tree = [np.where(tree_of_bit == t)[0] for t in range(n_trees)]
    width = max(1, max(len(p) for p in per_tree))
    idx = np.zeros((n_trees, width), dtype=np.int64)
    mask = np.zeros((n_trees, width), dtype=bool)
    for t, cols in enumerate(per_tree):
        idx[t, :len(cols)] = cols
        mask[t, :len(cols)] = True
    return idx, mask


class TreeTokenTransformer(nn.Module):
    """One token per tree, then a transformer encoder over the 100 tokens."""

    def __init__(self, idx, mask, out_dim, d_token=64, n_blocks=3, n_heads=8,
                 dropout=0.1):
        super().__init__()
        self.register_buffer("idx", torch.from_numpy(idx))
        self.register_buffer("bitmask", torch.from_numpy(mask.astype(np.float32)))
        n_trees, width = idx.shape
        # Shared across trees: every tree is the same kind of object.
        self.proj = nn.Linear(width, d_token)
        # ...but the model may still want to tell them apart.
        self.tree_emb = nn.Parameter(torch.randn(1, n_trees, d_token) * 0.02)
        self.cls = nn.Parameter(torch.randn(1, 1, d_token) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d_token, nhead=n_heads, dim_feedforward=d_token * 2,
            dropout=dropout, activation="gelu", batch_first=True,
            norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_blocks)
        self.norm = nn.LayerNorm(d_token)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(d_token, out_dim)

    def forward(self, x):                      # x: (B, n_kept_bits)
        B = x.shape[0]
        g = x[:, self.idx.reshape(-1)].view(B, *self.idx.shape)   # (B, trees, width)
        g = g * self.bitmask                                       # kill padding
        tok = self.proj(g) + self.tree_emb                         # (B, trees, d)
        tok = torch.cat([self.cls.expand(B, -1, -1), tok], dim=1)
        z = self.encoder(tok)
        return self.head(self.drop(self.norm(z[:, 0])))


SIZES = {
    "mlp":       {"small": (128, 64), "medium": (512, 256), "large": (1024, 512, 256)},
    "tabresnet": {"small": dict(d=64, d_hidden=128, n_blocks=2),
                  "medium": dict(d=192, d_hidden=384, n_blocks=3),
                  "large": dict(d=384, d_hidden=768, n_blocks=4)},
    "treetf":    {"small": dict(d_token=32, n_blocks=2, n_heads=4),
                  "medium": dict(d_token=64, n_blocks=3, n_heads=8),
                  "large": dict(d_token=128, n_blocks=4, n_heads=8)},
    # Bit tokens are far more numerous, so these stay deliberately narrow and
    # shallow; the cost is driven by the token count, not by the width.
    "bittf":     {"small": dict(d_token=32, n_blocks=1, n_heads=4),
                  "medium": dict(d_token=48, n_blocks=2, n_heads=6),
                  "large": dict(d_token=64, n_blocks=3, n_heads=8)},
}


def build_model(head, size, n_bits, n_classes, dropout, idx=None, mask=None):
    if head == "mlp":
        return MLPHead(n_bits, n_classes, hidden_dims=SIZES["mlp"][size],
                       dropout=dropout, batchnorm=True)
    if head == "tabresnet":
        return TabResNet(n_bits, n_classes, dropout=dropout, **SIZES["tabresnet"][size])
    if head == "treetf":
        return TreeTokenTransformer(idx, mask, n_classes, dropout=dropout,
                                    **SIZES["treetf"][size])
    if head == "bittf":
        return BitTokenTransformer(n_bits, n_classes, dropout=dropout,
                                   **SIZES["bittf"][size])
    raise ValueError(head)


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #
def metrics(y, proba):
    p = np.clip(proba, 1e-7, 1 - 1e-7)
    return {"auc": float(roc_auc_score(y, p[:, 1])),
            "accuracy": float((p.argmax(1) == y).mean()),
            "logloss": float(log_loss(y, p, labels=[0, 1]))}


@torch.no_grad()
def predict(model, X, device, bs=4096):
    model.eval()
    out = []
    for i in range(0, len(X), bs):
        xb = torch.from_numpy(X[i:i + bs]).to(device)
        out.append(torch.softmax(model(xb), dim=1).cpu().numpy())
    return np.concatenate(out)


# --------------------------------------------------------------------------- #
# cached encoding
# --------------------------------------------------------------------------- #
def build_encoding(args):
    """Fit the forest and build the bits, caching the result so a second
    session (or a second run in the same grid) reuses it instead of refitting."""
    tag = (f"t{args.task}_s{args.seed}_r{args.rf_trees}_d{args.rf_depth}"
           f"_l{args.rf_min_leaf}_m{args.max_rows}_{args.encoding}")
    cache = os.path.join(args.cache_dir, f"enc_{tag}.npz") if args.cache_dir else None
    ds = load_task(args.task, seed=args.seed, max_rows=args.max_rows)
    if cache and os.path.exists(cache):
        z = np.load(cache, allow_pickle=True)
        print(f"[cache] encoding loaded from {cache}", flush=True)
        rf = None
        return ds, z["clean_train"], z["val"], z["test"], z["depths"], z["counts"], rf
    rf = RandomForestClassifier(n_estimators=args.rf_trees, max_depth=args.rf_depth,
                                min_samples_leaf=args.rf_min_leaf, n_jobs=-1,
                                random_state=args.seed).fit(ds.X_train, ds.y_train)
    hard = encode_margins(split_margins(rf, ds.X_train), 0.0).astype(np.float32)
    val = split_direction_encoding(rf, ds.X_val).astype(np.float32)
    test = split_direction_encoding(rf, ds.X_test).astype(np.float32)
    depths, _ = node_depths_and_samples(rf)
    counts = np.array([int((e.tree_.feature >= 0).sum()) for e in rf.estimators_])
    clean = hard
    if args.encoding == "oob":
        ms, mean_oob = oob_mask_matrix(rf, len(ds.y_train))
        clean = hard * ms
        print(f"[bits] out-of-bag honest: each training row keeps bits from "
              f"{mean_oob:.1f}/{args.rf_trees} trees", flush=True)
    if cache:
        os.makedirs(args.cache_dir, exist_ok=True)
        np.savez_compressed(cache, clean_train=clean, val=val, test=test,
                            depths=depths, counts=counts)
        print(f"[cache] encoding saved to {cache}", flush=True)
    return ds, clean, val, test, depths, counts, rf


def cached_ceiling(ds, args):
    cache = (os.path.join(args.cache_dir, f"ceiling_t{args.task}_s{args.seed}"
                          f"_m{args.max_rows}.json") if args.cache_dir else None)
    if cache and os.path.exists(cache):
        c = json.load(open(cache))
        print(f"[cache] tree baselines loaded from {cache}", flush=True)
        return c
    c = {}
    for name in ("xgboost", "lightgbm", "random_forest"):
        te, _, _ = fit_tree_baseline(name, ds, {"seed": args.seed})
        c[name] = te["auc"]
        print(f"  {name:14s} ranking score = {te['auc']:.4f}", flush=True)
    if cache:
        os.makedirs(args.cache_dir, exist_ok=True)
        json.dump(c, open(cache, "w"), indent=1)
    return c


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", type=int, default=361065, help="default: MagicTelescope")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-rows", type=int, default=16000)
    ap.add_argument("--rf-trees", type=int, default=100)
    ap.add_argument("--rf-depth", type=int, default=6)
    ap.add_argument("--rf-min-leaf", type=int, default=5)
    ap.add_argument("--encoding", default="oob", choices=["oob", "infold"])
    ap.add_argument("--drop-layers", type=str, default=None)
    ap.add_argument("--keep-layers", type=str, default=None)
    # the head
    ap.add_argument("--head", default="linear",
                    choices=["linear", "mlp", "tabresnet", "bittf", "treetf"])
    ap.add_argument("--size", default="medium", choices=["small", "medium", "large"])
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--weight-decay", type=float, default=1e-5)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--patience", type=int, default=40,
                    help="stop when validation has not improved for this many epochs")
    ap.add_argument("--C", type=float, nargs="+",
                    default=[1e-4, 3e-4, 1e-3, 3e-3, 0.01, 0.03, 0.1, 0.3, 1.0],
                    help="penalties swept for the linear reference; best on validation")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--amp", default="auto", choices=["auto", "on", "off"],
                    help="mixed precision. Attention over bit tokens is "
                         "compute-bound and a T4's tensor cores only engage in "
                         "half precision, so this is on by default on a GPU")
    ap.add_argument("--time-probe", type=int, default=0,
                    help="train this many epochs, report seconds per epoch and "
                         "the projected full-run time, then stop")
    # resuming
    ap.add_argument("--cache-dir", default=None,
                    help="where the forest, bits and tree baselines are cached "
                         "(point at Google Drive to survive a disconnect)")
    ap.add_argument("--ckpt-dir", default=None, help="training checkpoints")
    ap.add_argument("--ckpt-every", type=int, default=25)
    ap.add_argument("--fresh", action="store_true", help="ignore any checkpoint")
    ap.add_argument("--label", default=None)
    ap.add_argument("--out", default="results/deep_head")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    label = args.label or f"{args.head}_{args.size}_do{args.dropout}_wd{args.weight_decay}"

    dev = args.device
    if dev == "auto":
        dev = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(dev)

    ds, clean_train, T_val, T_test, depths, counts, rf = build_encoding(args)
    n_bits_all = len(depths)

    kept, what = np.ones(n_bits_all, dtype=bool), "all depths"
    if args.keep_layers:
        k = [int(v) for v in args.keep_layers.split(",")]
        kept, what = np.isin(depths, k), f"keep only depths {k}"
    elif args.drop_layers:
        d = [int(v) for v in args.drop_layers.split(",")]
        kept, what = ~np.isin(depths, d), f"drop depths {d}"
    Xtr, Xva, Xte = clean_train[:, kept], T_val[:, kept], T_test[:, kept]
    n_bits = int(kept.sum())
    print(f"\n=== {ds.name} | head = {args.head} ({args.size}) | {what} "
          f"-> {n_bits} bits ===", flush=True)
    print(f"[data] train={len(ds.y_train)} val={len(ds.y_val)} test={len(ds.y_test)}",
          flush=True)

    print("[ceiling] tree baselines on the raw features ...", flush=True)
    ceil = cached_ceiling(ds, args)
    best_tree = max(ceil, key=ceil.get)

    t0 = time.time()
    hist = []

    # ---------------- the linear reference ----------------
    if args.head == "linear":
        best = None
        for C in args.C:
            lr = LogisticRegression(C=C, max_iter=3000).fit(Xtr, ds.y_train)
            m = {s: metrics(y, lr.predict_proba(X)) for s, X, y in
                 [("train", Xtr, ds.y_train), ("val", Xva, ds.y_val), ("test", Xte, ds.y_test)]}
            print(f"    C={C:<7g} train {m['train']['auc']:.4f}  val {m['val']['auc']:.4f}"
                  f"  test {m['test']['auc']:.4f}", flush=True)
            if best is None or m["val"]["auc"] > best["val"]["auc"]:
                best, best_C = m, C
        res = dict(best, n_params=int(Xtr.shape[1] + 1), C=best_C, epochs_run=0,
                   best_epoch=0)
        if best_C in (min(args.C), max(args.C)):
            print(f"    [!] the chosen penalty {best_C:g} sits at the edge of the "
                  f"grid; widen --C to be sure", flush=True)
    # ---------------- the deep heads ----------------
    else:
        torch.manual_seed(args.seed); np.random.seed(args.seed)
        idx = msk = None
        if args.head == "treetf":
            if rf is None:
                # cache hit: rebuild the grouping from the cached per-tree counts
                tree_of_bit = np.repeat(np.arange(len(counts)), counts)[kept]
                n_trees = len(counts)
                per = [np.where(tree_of_bit == t)[0] for t in range(n_trees)]
                width = max(1, max(len(p) for p in per))
                idx = np.zeros((n_trees, width), dtype=np.int64)
                msk = np.zeros((n_trees, width), dtype=bool)
                for t, cols in enumerate(per):
                    idx[t, :len(cols)] = cols; msk[t, :len(cols)] = True
            else:
                idx, msk = tree_token_index(rf, kept)
            print(f"[head] {idx.shape[0]} tree tokens, up to {idx.shape[1]} bits each",
                  flush=True)
        model = build_model(args.head, args.size, n_bits, 2, args.dropout, idx, msk).to(device)
        n_par = sum(p.numel() for p in model.parameters())
        print(f"[head] {args.head} {args.size}: {n_par:,} parameters | "
              f"dropout {args.dropout} | weight decay {args.weight_decay}", flush=True)
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                weight_decay=args.weight_decay)
        use_amp = (args.amp == "on") or (args.amp == "auto" and device.type == "cuda")
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
        if use_amp:
            print("[head] mixed precision on", flush=True)
        loader = DataLoader(TensorDataset(torch.from_numpy(Xtr),
                                          torch.from_numpy(ds.y_train)),
                            batch_size=args.batch_size, shuffle=True, drop_last=False)

        start_ep, best_val, best_state, best_ep, bad = 1, -1.0, None, 0, 0
        ck = os.path.join(args.ckpt_dir, f"ck_{ds.name}_{label}.pt") if args.ckpt_dir else None
        if ck and os.path.exists(ck) and not args.fresh:
            try:
                st = torch.load(ck, map_location=device, weights_only=False)
                model.load_state_dict(st["model"]); opt.load_state_dict(st["opt"])
                start_ep, best_val, best_ep = st["epoch"] + 1, st["best_val"], st["best_ep"]
                best_state, hist, bad = st["best_state"], st["hist"], st["bad"]
                torch.set_rng_state(st["rng"].cpu() if torch.is_tensor(st["rng"]) else st["rng"])
                print(f"[resume] continuing from epoch {start_ep} "
                      f"(best validation {best_val:.4f} at epoch {best_ep})", flush=True)
            except Exception as exc:  # noqa: BLE001 - a truncated file is not fatal
                print(f"[resume] checkpoint unreadable ({exc}); starting fresh", flush=True)

        for ep in range(start_ep, args.epochs + 1):
            model.train()
            tot = 0.0
            for xb, yb in loader:
                xb, yb = xb.to(device), yb.to(device)
                opt.zero_grad(set_to_none=True)
                with torch.autocast("cuda", enabled=use_amp):
                    loss = F.cross_entropy(model(xb), yb)
                scaler.scale(loss).backward()
                scaler.step(opt); scaler.update()
                tot += float(loss) * len(xb)
            if args.time_probe and ep == args.time_probe:
                per = (time.time() - t0) / args.time_probe
                print(f"\n[probe] {per:.1f} s per epoch at batch {args.batch_size} "
                      f"on {n_bits} bits", flush=True)
                print(f"[probe] a {args.epochs}-epoch run would take about "
                      f"{per * args.epochs / 60:.0f} min "
                      f"(less if it stops early)", flush=True)
                return
            m = {s: metrics(y, predict(model, X, device)) for s, X, y in
                 [("train", Xtr, ds.y_train), ("val", Xva, ds.y_val)]}
            hist.append(dict(epoch=ep, train_loss=tot / len(Xtr),
                             train_auc=m["train"]["auc"], val_auc=m["val"]["auc"],
                             train_acc=m["train"]["accuracy"], val_acc=m["val"]["accuracy"],
                             gap=m["train"]["auc"] - m["val"]["auc"]))
            if m["val"]["auc"] > best_val:
                best_val, best_ep, bad = m["val"]["auc"], ep, 0
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            else:
                bad += 1
            if ep % max(1, args.epochs // 10) == 0 or ep == 1:
                print(f"    epoch {ep:4d}/{args.epochs}  train {m['train']['auc']:.4f}"
                      f"  val {m['val']['auc']:.4f}  (best {best_val:.4f} @ {best_ep})",
                      flush=True)
            if ck and (ep % args.ckpt_every == 0 or ep == args.epochs):
                tmp = ck + ".tmp"
                torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), epoch=ep,
                                best_val=best_val, best_ep=best_ep, best_state=best_state,
                                hist=hist, bad=bad, rng=torch.get_rng_state()), tmp)
                os.replace(tmp, ck)          # never leave a half-written checkpoint
            if bad >= args.patience:
                print(f"    stopped early at epoch {ep}: no improvement for "
                      f"{args.patience} epochs", flush=True)
                break

        model.load_state_dict(best_state)
        res = {s: metrics(y, predict(model, X, device)) for s, X, y in
               [("train", Xtr, ds.y_train), ("val", Xva, ds.y_val), ("test", Xte, ds.y_test)]}
        res.update(n_params=int(n_par), epochs_run=len(hist), best_epoch=best_ep)
        if ck and os.path.exists(ck):
            os.remove(ck)                    # finished: the checkpoint is dead weight

    res["gap"] = res["train"]["auc"] - res["val"]["auc"]
    res["seconds"] = round(time.time() - t0, 1)

    print(f"\n  {'split':6s} {'ranking':>9s} {'accuracy':>9s} {'log loss':>9s}")
    for s in ("train", "val", "test"):
        print(f"  {s:6s} {res[s]['auc']:9.4f} {res[s]['accuracy']:9.4f} "
              f"{res[s]['logloss']:9.4f}" + ("   <- clean bits" if s == "train" else ""))
    print(f"\n  overfitting gap (train minus validation ranking score): {res['gap']:+.4f}")
    print(f"  best tree on raw features: {ceil[best_tree]:.4f} ({best_tree.replace('_',' ')})")
    print(f"  test minus best tree: {res['test']['auc'] - ceil[best_tree]:+.4f}")

    summary = dict(dataset=ds.name, task=args.task, label=label, head=args.head,
                   size=args.size if args.head != "linear" else "-",
                   dropout=args.dropout, weight_decay=args.weight_decay, lr=args.lr,
                   batch_size=args.batch_size, epochs=args.epochs,
                   patience=args.patience, seed=args.seed, encoding=args.encoding,
                   bits_selection=what, n_bits=n_bits, n_bits_all=int(n_bits_all),
                   n_train=int(len(ds.y_train)), n_val=int(len(ds.y_val)),
                   n_test=int(len(ds.y_test)), ceiling=ceil, best_tree=best_tree,
                   tree_ceiling=ceil[best_tree], result=res, history=hist)
    path = os.path.join(args.out, f"deephead_{ds.name}.json")
    json.dump(summary, open(path, "w"), indent=2)
    print(f"\n[deep-head] summary -> {path}")


if __name__ == "__main__":
    main()
