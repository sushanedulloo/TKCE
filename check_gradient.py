"""Is the gradient dead? — decide whether MORE EPOCHS can still change anything.

Loads a saved checkpoint and measures, on the FULL training set:

  * ||g||        the global gradient norm (how strong the learning signal still is)
  * ||theta||    the weight norm (does it keep growing? -> the overconfidence mechanism)
  * per-layer gradient norms (is one part dead while another still learns?)
  * how many training samples still produce a meaningful gradient
  * a projected upper bound on how far the weights could still move in N more epochs

If ||g|| is essentially zero, training longer is a waste of GPU time, and that is a
concrete answer rather than a guess.

    python check_gradient.py --ckpt /content/drive/MyDrive/tkce_double_descent/ckpt_credit_x+tree_ep004000.pt
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F

from run_fusion import (FusionModel, fit_encoder_rf, oob_honest_encoding,
                        split_direction_encoding)
from tkce.data import load_task
from tkce.train import resolve_device


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="path to a ckpt_*.pt file")
    ap.add_argument("--task", type=int, default=361055)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-rows", type=int, default=16000)
    ap.add_argument("--views", default="", help="override; default: read from checkpoint")
    ap.add_argument("--encoding", default="oob", choices=["infold", "oob"])
    ap.add_argument("--tau", type=float, default=0.0)
    ap.add_argument("--rf-trees", type=int, default=100)
    ap.add_argument("--rf-depth", type=int, default=6)
    ap.add_argument("--rf-min-leaf", type=int, default=5)
    ap.add_argument("--fusion", default="tabresnet")
    ap.add_argument("--feat-dim", type=int, default=128)
    ap.add_argument("--ext-d", type=int, default=192)
    ap.add_argument("--ext-d-hidden", type=int, default=256)
    ap.add_argument("--ext-blocks", type=int, default=3)
    ap.add_argument("--d", type=int, default=256)
    ap.add_argument("--d-hidden", type=int, default=512)
    ap.add_argument("--n-blocks", type=int, default=4)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--future-epochs", type=int, default=11000,
                    help="how many more epochs you are considering running")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    device = resolve_device(args.device)
    try:
        ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    except TypeError:
        ck = torch.load(args.ckpt, map_location="cpu")
    cfg = ck.get("cfg", {})
    views = (args.views or cfg.get("views", "x+tree")).split("+")
    epoch = ck["epoch"]
    print(f"\n=== GRADIENT HEALTH CHECK ===")
    print(f"checkpoint : {os.path.basename(args.ckpt)}")
    print(f"epoch      : {epoch}   views: {'+'.join(views)}   device: {device}\n")

    ds = load_task(args.task, seed=args.seed, max_rows=args.max_rows)
    rf = fit_encoder_rf(ds.X_train, ds.y_train, args)
    enc_tr = split_direction_encoding(rf, ds.X_train, args.tau)
    if args.encoding == "oob":
        enc_tr, _ = oob_honest_encoding(rf, enc_tr, len(ds.y_train))

    model = FusionModel(ds.n_features, enc_tr.shape[1], ds.n_classes, views, args).to(device)
    model.load_state_dict(ck["model"])
    model.eval()                      # no dropout noise in the measurement

    X = torch.from_numpy(ds.X_train).to(device)
    T = torch.from_numpy(enc_tr).to(device)
    y = torch.from_numpy(ds.y_train).to(device)

    # ---- full-batch gradient ----
    model.zero_grad(set_to_none=True)
    n, bs, tot = len(y), 4096, 0.0
    for s in range(0, n, bs):
        out = model(X[s:s+bs], T[s:s+bs])
        loss = F.cross_entropy(out, y[s:s+bs], reduction="sum") / n
        loss.backward()
        tot += loss.item()
    gnorm = torch.sqrt(sum((p.grad**2).sum() for p in model.parameters()
                           if p.grad is not None)).item()
    wnorm = torch.sqrt(sum((p**2).sum() for p in model.parameters())).item()

    print(f"train loss (full batch)   : {tot:.6f}")
    print(f"GRADIENT NORM  ||g||      : {gnorm:.3e}")
    print(f"weight norm    ||theta||  : {wnorm:.3f}")
    print(f"relative       ||g||/||w||: {gnorm/wnorm:.3e}")

    # ---- per-layer ----
    print("\nper-parameter-tensor gradient norms (largest 8):")
    rows = sorted(((p.grad.norm().item(), nm, tuple(p.shape))
                   for nm, p in model.named_parameters() if p.grad is not None),
                  reverse=True)
    for g, nm, sh in rows[:8]:
        print(f"   {g:.3e}   {nm:52s} {sh}")
    dead = sum(1 for g, _, _ in rows if g < 1e-8)
    print(f"   ({dead} of {len(rows)} tensors have a gradient below 1e-8)")

    # ---- which samples still push? ----
    with torch.no_grad():
        losses = []
        for s in range(0, n, bs):
            out = model(X[s:s+bs], T[s:s+bs])
            losses.append(F.cross_entropy(out, y[s:s+bs], reduction="none").cpu())
        L = torch.cat(losses).numpy()
    print(f"\nper-sample training loss: median {np.median(L):.2e}  "
          f"mean {L.mean():.2e}  max {L.max():.3f}")
    for thr in (1e-2, 1e-3, 1e-4):
        print(f"   samples with loss > {thr:<7}: {int((L > thr).sum()):5d} / {n} "
              f"({(L > thr).mean()*100:5.2f}%)  <- these are the only ones still teaching")

    # ---- projection ----
    steps = int(np.ceil(n / args.batch_size)) * args.future_epochs
    move = args.lr * gnorm * steps
    print(f"\n=== WOULD {args.future_epochs} MORE EPOCHS CHANGE ANYTHING? ===")
    print(f"  steps in {args.future_epochs} epochs        : {steps:,}")
    print(f"  upper-bound weight movement    : lr x ||g|| x steps = "
          f"{args.lr:g} x {gnorm:.2e} x {steps:,} = {move:.4f}")
    print(f"  as a fraction of ||theta||     : {move/wnorm*100:.2f}%")
    print("  (upper bound: real movement is smaller because gradient directions cancel)")
    if move / wnorm < 0.01:
        print("\n  VERDICT: gradient is effectively DEAD — more epochs will not change the model.")
    elif move / wnorm < 0.10:
        print("\n  VERDICT: gradient is very weak — expect only small changes.")
    else:
        print("\n  VERDICT: gradient is still meaningful — more training can still move the model.")


if __name__ == "__main__":
    main()
