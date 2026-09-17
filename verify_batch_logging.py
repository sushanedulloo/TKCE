"""Prove that --log-batches does not change the training at all.

Runs run_fusion.py twice with the same seed — once plain, once with
--log-batches — and checks that

  1. every per-epoch number is BIT-IDENTICAL between the two runs
     (train/val loss, train/val AUC, accuracy, for every epoch and member), and
  2. the recorded per-batch numbers are complete and internally consistent
     (one row per batch per epoch per member per view; run_avg_loss really is
     the running mean of batch_loss inside each epoch).

Logging only reads the loss tensor and the logits that training already
computed, under torch.no_grad(), after opt.step(). It consumes no randomness,
adds no forward pass, and never touches a parameter or a BatchNorm statistic —
so the two runs must agree exactly. This script is the check, not the claim.

    python verify_batch_logging.py                     # quick CPU check
    python verify_batch_logging.py --device auto --epochs 6
"""

from __future__ import annotations

import argparse
import glob
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd

EPOCH_COLS = ["train_loss", "val_loss", "train_auc", "val_auc",
              "train_acc", "val_acc"]


def _run(out_dir, extra, common):
    cmd = [sys.executable, "-u", "run_fusion.py", "--out", out_dir] + common + extra
    print("  $ " + " ".join(cmd[2:]), flush=True)
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-3000:], r.stderr[-3000:])
        raise SystemExit(f"run failed: {out_dir}")
    return r.stdout


def _epochs(out_dir):
    f = glob.glob(os.path.join(out_dir, "fusion_*_epochs.csv"))
    if not f:
        raise SystemExit(f"no epochs csv in {out_dir}")
    return pd.read_csv(f[0])


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", type=int, default=361055)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--max-rows", type=int, default=2000)
    ap.add_argument("--rf-trees", type=int, default=20)
    ap.add_argument("--ensemble", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--views", default="x;x+tree")
    ap.add_argument("--keep", action="store_true", help="keep the two run dirs")
    args = ap.parse_args()

    common = ["--task", str(args.task), "--views", args.views,
              "--encoding", "oob", "--epochs", str(args.epochs),
              "--ensemble", str(args.ensemble), "--max-rows", str(args.max_rows),
              "--rf-trees", str(args.rf_trees), "--batch-size", str(args.batch_size),
              "--dropout", "0", "--l1", "0", "--lr", "3e-4",
              "--seed", "0", "--device", args.device]

    tmp = tempfile.mkdtemp(prefix="verify_batchlog_")
    plain, logged = os.path.join(tmp, "plain"), os.path.join(tmp, "logged")
    try:
        print("1) training WITHOUT --log-batches")
        _run(plain, [], common)
        print("2) training WITH --log-batches (same seed)")
        _run(logged, ["--log-batches"], common)

        a, b = _epochs(plain), _epochs(logged)
        fails = []

        # ---- 1. the training must be identical ----
        if a.shape != b.shape:
            fails.append(f"epoch tables differ in shape: {a.shape} vs {b.shape}")
        else:
            key = ["model", "epoch"] + (["member"] if "member" in a.columns else [])
            a_s = a.sort_values(key).reset_index(drop=True)
            b_s = b.sort_values(key).reset_index(drop=True)
            worst = {}
            for c in EPOCH_COLS:
                if c not in a_s.columns:
                    continue
                d = (a_s[c].to_numpy(float) - b_s[c].to_numpy(float))
                worst[c] = float(np.max(np.abs(d)))
                if worst[c] != 0.0:
                    fails.append(f"{c}: max |difference| = {worst[c]:.3e} (must be 0)")
            print("\n   per-epoch agreement (max absolute difference over all "
                  "epochs and members):")
            for c, v in worst.items():
                print(f"     {c:11s} {v:.3e}   {'identical' if v == 0 else 'DIFFERS'}")

        # ---- 2. the batch record must be complete and consistent ----
        bf = glob.glob(os.path.join(logged, "fusion_*_batches.csv"))
        if not bf:
            fails.append("no batches csv was written")
        else:
            bd = pd.read_csv(bf[0])
            n_views = len(args.views.split(";"))
            groups = bd.groupby(["model", "member", "epoch"])
            expected_groups = n_views * args.ensemble * args.epochs
            print(f"\n   batch rows: {len(bd):,} in {groups.ngroups} "
                  f"(model, member, epoch) groups; expected {expected_groups}")
            if groups.ngroups != expected_groups:
                fails.append(f"expected {expected_groups} groups, got {groups.ngroups}")
            sizes = groups.size().unique()
            print(f"   batches per epoch: {sorted(sizes)}  (one row per batch)")
            for (m, mem, ep), g in groups:
                g = g.sort_values("batch")
                if list(g["batch"]) != list(range(1, len(g) + 1)):
                    fails.append(f"batch numbers not 1..n for {m}/{mem}/epoch {ep}")
                    break
                run_mean = g["batch_loss"].expanding().mean().to_numpy()
                if not np.allclose(run_mean, g["run_avg_loss"].to_numpy(), atol=2e-6):
                    fails.append(f"run_avg_loss is not the running mean "
                                 f"({m}/{mem}/epoch {ep})")
                    break
            print("   run_avg_loss == running mean of batch_loss within each epoch: ok"
                  if not any("run_avg" in f for f in fails) else "   run_avg_loss: WRONG")
            nan_auc = bd["batch_auc"].isna().sum()
            print(f"   batch_auc present for {len(bd) - nan_auc:,}/{len(bd):,} rows "
                  f"({nan_auc} single-class batches -> blank, as intended)")
            if "batch_total_loss" in bd.columns:
                same = np.allclose(bd["batch_loss"], bd["batch_total_loss"], atol=1e-9)
                print(f"   batch_total_loss == batch_loss with --l1 0: "
                      f"{'ok' if same else 'DIFFERS (unexpected at l1=0)'}")
                if not same:
                    fails.append("batch_total_loss != batch_loss although --l1 0")

        # ---- verdict ----
        print()
        if fails:
            print("FAILED:")
            for f in fails:
                print("  -", f)
            raise SystemExit(1)
        print("PASSED — --log-batches changed nothing: every per-epoch number is "
              "bit-identical to the plain run, and the per-batch record is "
              "complete and consistent.")
    finally:
        if args.keep:
            print(f"\nrun dirs kept in {tmp}")
        else:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
