"""Fingerprint a run_fusion.py output directory and compare it to the reference.

When a run "looks different from last time", the cause is almost always one of
four things, and each leaves a fingerprint in the saved JSON:

  * a different ARM        -> the parameter count / trunk size differs
  * different DATA         -> the tree ceiling or the split sizes differ
  * an accidental FLAG     -> a perturbation, dropout or L1 is switched on
  * a genuinely different TRAJECTORY -> arm, data and flags all match, and the
                              train/val curves still differ

This prints all four, side by side with the reference values for credit at full
capacity, so the answer is a single glance rather than a discussion.

    python diagnose_run.py results/fusion/cap_1_full
    python diagnose_run.py results/fusion/cap_*            # several at once
"""

from __future__ import annotations

import glob
import json
import os
import sys

import pandas as pd

# Reference: credit (task 361055), x+tree, OOB-honest, full-capacity trunk,
# dropout 0 / L1 0, lr 3e-4, batch 128 — the configuration that produced
# 0.845-0.848 test AUC in five independent runs.
REF = {
    "dataset": "credit",
    "task": 361055,
    "n_train": 11200, "n_val": 2400, "n_test": 2400,
    "tree_ceiling": 0.8453,
    "encoding": "oob",
    "enc_width": 5400,          # varies by a bit or two with the forest fit
    "params": 2439682,
    "d": 256, "d_hidden": 512, "n_blocks": 4,
    "test_auc_lo": 0.843, "test_auc_hi": 0.851,
}
PERTURBATIONS = ["tau", "enc_noise", "deep_flip_p", "deep_delete",
                 "deep_layers", "flip_ramp", "flip_uniform", "dropout", "l1"]


def _mark(ok):
    return "ok    " if ok else "DIFFERS"


def fingerprint(run_dir):
    js = [f for f in glob.glob(os.path.join(run_dir, "fusion_*.json"))
          if "epochs" not in f and "batches" not in f]
    if not js:
        print(f"\n=== {run_dir} ===\n  no fusion_*.json — run not finished?")
        return
    s = json.load(open(js[0]))
    ep = glob.glob(os.path.join(run_dir, "fusion_*_epochs.csv"))
    e = pd.read_csv(ep[0]) if ep else None

    print(f"\n=== {run_dir} ===")

    # ---- data ----
    print("  DATA")
    print(f"    dataset / task        {s.get('dataset')} / {s.get('task')}"
          f"    [{_mark(s.get('dataset') == REF['dataset'] and s.get('task') == REF['task'])}]")
    ceil = s.get("tree_ceiling")
    print(f"    tree ceiling          {ceil:.4f}   reference {REF['tree_ceiling']:.4f}"
          f"    [{_mark(abs(ceil - REF['tree_ceiling']) < 0.002)}]"
          "   <- a different value here means the DATA or SPLIT changed")
    per_model = s.get("ceiling") or {}
    if per_model:
        print("      " + "  ".join(f"{k}={v:.4f}" for k, v in per_model.items()))
    print(f"    split                 {s.get('split')}")
    w = s.get("tree_encoding_width") or 0
    w_ok = abs(w - REF["enc_width"]) <= 0.05 * REF["enc_width"]
    print(f"    encoding / width      {s.get('encoding')} / {w} bits"
          f"    [{_mark(s.get('encoding') == REF['encoding'] and w_ok)}]"
          f"   (reference ~{REF['enc_width']}; a few bits either way is just the forest fit)")

    # ---- flags that would change the result ----
    print("  FLAGS (anything non-zero here is a perturbation or a regulariser)")
    active = [(k, s.get(k)) for k in PERTURBATIONS
              if s.get(k) not in (None, 0, 0.0, False, [], "")]
    print("    " + ("ALL ZERO — plain training" if not active
                    else "  ".join(f"{k}={v}" for k, v in active)))
    print(f"    lr={s.get('lr')}  batch={s.get('batch_size')}  "
          f"weight_decay={s.get('weight_decay')}  ensemble={s.get('ensemble')}")

    # ---- per view config ----
    print("  ARMS")
    for r in s.get("results", []):
        pa, dd = r.get("params"), r.get("d")
        pm = r.get("params_M")
        # runs made before the exact count was recorded still carry params_M
        if pa is None and pm is not None:
            pa_txt, is_full = f"~{pm:.2f}M", abs(pm - 2.44) < 0.005
        elif pa is None:
            pa_txt, is_full = "not recorded", None
        else:
            pa_txt, is_full = f"{pa:,d}", pa == REF["params"]
        line = (f"    {r['views']:<8s} params={pa_txt:>12s}  d={dd}  "
                f"d_hidden={r.get('d_hidden')}  blocks={r.get('n_blocks')}")
        if r["views"] == "x+tree" and is_full is not None:
            line += f"   [{'FULL capacity' if is_full else 'REDUCED capacity'}]"
        print(line)
        tr = fin = None
        if e is not None:
            g = e[e.model == r["model"]].groupby("epoch").train_auc.mean()
            hit = g[g >= 0.99]
            fin = g.iloc[-1]
            tr = int(hit.index.min()) if len(hit) else None
        print(f"             test_auc={r['test_auc']:.4f}  "
              f"best_val={r['best_val_auc']:.4f} @ epoch {r.get('best_epoch')}  "
              f"final_train_auc={fin if fin is None else round(fin, 4)}  "
              f"ep99={tr if tr is not None else 'never'}")
        if r["views"] == "x+tree" and is_full:
            inrange = REF["test_auc_lo"] <= r["test_auc"] <= REF["test_auc_hi"]
            print(f"             vs reference 0.845-0.848: "
                  f"[{_mark(inrange)}]"
                  f"{'' if inrange else '  <- full capacity but OUT of the usual band'}")
            if tr is None:
                print("             [DIFFERS] full capacity never reached train AUC 0.99 "
                      "— every previous run did, by epoch 5-7")


def main():
    args = sys.argv[1:]
    if not args:
        print(__doc__)
        raise SystemExit(1)
    dirs = []
    for a in args:
        dirs.extend(sorted(glob.glob(a)) if any(c in a for c in "*?[") else [a])
    for d in dirs:
        if os.path.isdir(d):
            fingerprint(d)
    print("\nPaste this whole output if the cause is still not obvious.")


if __name__ == "__main__":
    main()
