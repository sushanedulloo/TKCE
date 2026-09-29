"""Deep heads on MagicTelescope: the capacity-versus-generalisation experiment.

Same setup as the feature-regularisation suite, one thing changed: the head.
The question is whether a deep network's extra capacity turns into better
predictions once the features have been regularised, and whether the network
needs regularising of its own on top.

MagicTelescope was chosen because it is the binary dataset where the linear
head fell clearly short of the trees: 0.9047 against a ceiling of 0.9377, a gap
of 0.033, more than three times the measurement noise. Credit had no room left
(the linear head already matched the tree) and house_16H's shortfall of 0.0085
is inside the noise, so neither could show a gain.

    Part A   the linear reference, penalty chosen on validation
    Part B   3 deep heads x 3 sizes x 4 regularisation settings, SHALLOW bits
    Part C   the same heads and sizes on the FULL encoding, 2 settings

Part C is the capacity-versus-generalisation comparison: if big heads help on
the shallow bits but hurt on the full ones, then feature regularisation is what
lets a network use its capacity.

The suite is resumable at two levels. A finished run writes its own summary and
is skipped on a rerun, and within a run, training checkpoints every few epochs
so an interrupted run continues rather than restarting. The forest, the bits
and the tree baselines are cached, so the second session skips those too.

    python run_deep_head_suite.py --part B
    python run_deep_head_suite.py --aggregate-only --bundle
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys
import time

import numpy as np
import pandas as pd

TASK = 361065          # MagicTelescope
SHALLOW = "--drop-layers 4,5"
FULL = ""

HEADS = ["mlp", "tabresnet", "bittf"]

# Attention over bit tokens is quadratic in the number of bits, so the
# transformer gets a smaller batch (its activations are the (batch, bits,
# width) tokens) and, on the full encoding, fewer epochs and fewer arms. The
# other heads are cheap and keep the full grid.
BATCH = {"bittf": 64}
FULL_EPOCHS = {"bittf": 60}
FULL_SIZES = {"bittf": ["small", "medium"]}
SIZES = ["small", "medium", "large"]
# (label, hidden dropout, weight decay, INPUT dropout)
#
# The first grid was too weak to test anything. AdamW decay of 0.01 at lr 1e-3
# shrinks the weights by 1.5% over a whole run, so the "decay" arm moved the
# overfitting gap by 0.0002 -- it was a no-op dressed as a condition. And the
# heads' dropout only touches hidden activations, while 85-97% of the
# parameters, and so nearly all of the memorising, sit in the first layer.
#
# This grid fixes both: decay values that actually bind, and input dropout,
# which drops whole bits and is the only setting that reaches that first layer.
# It is the in-network analogue of the bit-flip noise that worked on the
# feature side.
REG = [("none",     0.0, 1e-5, 0.0),
       ("hidden",   0.3, 1e-5, 0.0),   # what the old grid called "dropout"
       ("decay",    0.0, 0.3,  0.0),   # decay that actually binds (36% shrinkage)
       ("input",    0.0, 1e-5, 0.3),   # the new lever, on its own
       ("input5",   0.0, 1e-5, 0.5),   # and harder
       ("all",      0.3, 0.3,  0.3)]   # everything together
REG_FULL = [r for r in REG if r[0] in ("none", "all")]

SHARED = ("--task 361065 --seed 0 --max-rows 16000 --rf-trees 100 --rf-depth 6 "
          "--rf-min-leaf 5 --encoding oob --lr 1e-3 --batch-size 256 "
          "--epochs 150 --patience 30 --device auto")
QUICK = "--max-rows 2500 --rf-trees 20 --epochs 15 --patience 15"
PY = sys.executable


def plan(part):
    """Every run in the grid: (bits label, extra bit flags, head, size, reg)."""
    jobs = []
    if part in ("A", "all"):
        jobs += [("shallow", SHALLOW, "linear", "-", ("-", 0.0, 0.0, 0.0)),
                 ("full", FULL, "linear", "-", ("-", 0.0, 0.0, 0.0))]
    if part in ("B", "all"):
        jobs += [("shallow", SHALLOW, h, s, r)
                 for h in HEADS for s in SIZES for r in REG]
    if part in ("C", "all"):
        jobs += [("full", FULL, h, s, r)
                 for h in HEADS for s in FULL_SIZES.get(h, SIZES) for r in REG_FULL]
    return jobs


def run_all(root, jobs, quick, ckpt_dir, cache_dir, sync_dir=None):
    print(f"[suite] {len(jobs)} runs -> {root}", flush=True)
    t_all = time.time()
    for i, (bits, flags, head, size, (rlab, do, wd, idr)) in enumerate(jobs, 1):
        label = (f"{head}" if head == "linear" else f"{head}_{size}_{rlab}")
        out = os.path.join(root, bits, label)
        done = glob.glob(os.path.join(out, "deephead_*.json"))
        if done:
            # Skip only if the finished run used the settings this grid asks for
            # now. Sizes and regularisation values have been revised once
            # already, and a label like "decay" meant weight decay 0.01 before
            # and 0.3 now; skipping on the name alone would silently mix the two.
            try:
                prev = json.load(open(done[0]))
                stale = [f"{k}: {prev.get(k)} -> {v}" for k, v in
                         [("size", size if head != "linear" else "-"),
                          ("dropout", do), ("weight_decay", wd),
                          ("input_dropout", idr)]
                         if head != "linear" and prev.get(k) != v]
            except Exception:  # noqa: BLE001 - unreadable file, just rerun it
                stale = ["unreadable"]
            if not stale:
                print(f"[suite] {i:3d}/{len(jobs)} {bits:8s} {label:24s} done, skipping",
                      flush=True)
                continue
            print(f"[suite] {i:3d}/{len(jobs)} {bits:8s} {label:24s} RERUN, settings "
                  f"changed ({'; '.join(stale)})", flush=True)
        cmd = [PY, "-u", "run_deep_head.py", "--head", head, "--label", label,
               "--out", out] + SHARED.split()
        if flags:
            cmd += flags.split()
        if head != "linear":
            cmd += ["--size", size, "--dropout", str(do), "--weight-decay", str(wd),
                    "--input-dropout", str(idr)]
            if head in BATCH:
                cmd += ["--batch-size", str(BATCH[head])]
            if bits == "full" and head in FULL_EPOCHS:
                cmd += ["--epochs", str(FULL_EPOCHS[head])]
            if ckpt_dir:
                cmd += ["--ckpt-dir", ckpt_dir, "--ckpt-every", "10"]
        if cache_dir:
            cmd += ["--cache-dir", cache_dir]
        if quick:
            cmd += QUICK.split()
        print(f"\n[suite] {i:3d}/{len(jobs)} {bits} :: {label}   "
              f"[{time.strftime('%H:%M:%S')}]", flush=True)
        os.makedirs(out, exist_ok=True)
        t0 = time.time()
        with open(os.path.join(out, "run.log"), "w") as lf:
            lf.write(f"# {bits} :: {label}\n# {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
                     f"# {' '.join(cmd)}\n\n")
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT, text=True, bufsize=1)
            for line in proc.stdout:
                sys.stdout.write(line)
                lf.write(line)
            proc.wait()
            lf.write(f"\n# exit {proc.returncode} after {time.time() - t0:.0f}s\n")
        ok = "ok" if proc.returncode == 0 else f"FAILED (exit {proc.returncode})"
        print(f"[suite] {label} -> {ok} in {time.time() - t0:.0f}s", flush=True)
        if sync_dir and proc.returncode == 0:
            # Copy this run to Drive immediately, so a disconnect costs at most
            # the run that was in flight, not everything since the last part.
            import shutil
            dst = os.path.join(sync_dir, bits, label)
            try:
                shutil.copytree(out, dst, dirs_exist_ok=True)
            except Exception as exc:  # noqa: BLE001 - Drive hiccup is not fatal
                print(f"[suite] could not sync to Drive: {exc}", flush=True)
    print(f"\n[suite] finished in {(time.time() - t_all) / 60:.1f} min", flush=True)


# --------------------------------------------------------------------------- #
def load_rows(root):
    rows = []
    for js in sorted(glob.glob(os.path.join(root, "*", "*", "deephead_*.json"))):
        s = json.load(open(js))
        r = s["result"]
        bits = os.path.basename(os.path.dirname(os.path.dirname(js)))
        lab = s["label"]
        reg = lab.split("_")[-1] if s["head"] != "linear" else "-"
        rows.append(dict(
            bits=bits, head=s["head"], size=s["size"], reg=reg, label=lab,
            n_bits=s["n_bits"], n_params=r["n_params"],
            dropout=s["dropout"], weight_decay=s["weight_decay"],
            train=r["train"]["auc"], val=r["val"]["auc"], test=r["test"]["auc"],
            gap=r["gap"], train_acc=r["train"]["accuracy"],
            test_acc=r["test"]["accuracy"], test_logloss=r["test"]["logloss"],
            best_epoch=r["best_epoch"], epochs_run=r["epochs_run"],
            seconds=r["seconds"], tree_ceiling=s["tree_ceiling"],
            best_tree=s["best_tree"]))
    return pd.DataFrame(rows)


def figures(t, root):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    BLUE, ORANGE, AQUA, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7"
    INK, MUTED, FAINT = "#0b0b0b", "#52514e", "#c8c8c4"
    plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": .22,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "figure.facecolor": "white", "legend.frameon": False})
    CEIL = t.tree_ceiling.iloc[0]
    lin = t[(t["head"] == "linear") & (t.bits == "shallow")]
    LIN = float(lin.test.iloc[0]) if len(lin) else np.nan
    HCOL = {"mlp": BLUE, "tabresnet": ORANGE, "bittf": AQUA, "treetf": VIOLET}

    # --- 1: test score by head, size and regularisation (shallow bits) ---
    sh = t[(t.bits == "shallow") & (t["head"] != "linear")]
    if len(sh):
        fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
        for a_, col, ttl, ref in [
                (ax[0], "test", "Test ranking score", LIN),
                (ax[1], "gap", "Overfitting gap (train minus validation)", 0.0)]:
            x = 0; ticks, labels = [], []
            for h in HEADS:
                for s in SIZES:
                    g = sh[(sh["head"] == h) & (sh["size"] == s)]
                    for j, (rl, *_rest) in enumerate(REG):
                        v = g[g.reg == rl]
                        if len(v):
                            a_.scatter([x], [float(v[col].iloc[0])], s=46,
                                       color=HCOL[h],
                                       alpha=0.3 + 0.7 * j / max(1, len(REG) - 1),
                                       edgecolor="white", lw=.8, zorder=3)
                    ticks.append(x); labels.append(f"{h}\n{s}")
                    x += 1
                x += 0.6
            if not np.isnan(ref):
                a_.axhline(ref, color=INK, lw=1.3, ls=":",
                           label=("linear head" if col == "test" else "no overfitting"))
            if col == "test":
                a_.axhline(CEIL, color=MUTED, lw=1.3, ls="--", label="best tree")
            a_.set_xticks(ticks); a_.set_xticklabels(labels, fontsize=7.5)
            a_.set_title(ttl, fontweight="bold", loc="left", fontsize=10)
            a_.legend(fontsize=8)
        ax[0].scatter([], [], s=46, color=MUTED, alpha=.35, label="no regularisation")
        ax[0].scatter([], [], s=46, color=MUTED, alpha=1.0, label="dropout + weight decay")
        ax[0].legend(fontsize=8, ncol=2)
        fig.suptitle("Deep heads on the shallow encoding: shade = regularisation strength",
                     fontweight="bold", fontsize=10.5, x=0.01, ha="left")
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        fig.savefig(os.path.join(root, "fig1_heads.png"), dpi=150)
        plt.close(fig)

    # --- 2: capacity versus generalisation, shallow against full ---
    d = t[t["head"] != "linear"]
    if len(d) and d.bits.nunique() > 1:
        fig, ax = plt.subplots(1, 2, figsize=(12, 4.4))
        for a_, col, ttl in [(ax[0], "test", "Test ranking score"),
                             (ax[1], "gap", "Overfitting gap")]:
            for bits, mk, ls in [("shallow", "o", "-"), ("full", "s", "--")]:
                g = d[d.bits == bits].sort_values("n_params")
                for h in HEADS:
                    gg = g[g["head"] == h]
                    if len(gg):
                        a_.plot(gg.n_params, gg[col], mk, ls=ls, ms=6, lw=1.2,
                                color=HCOL[h], alpha=.85,
                                label=f"{h}, {bits}" if col == "test" else None)
            a_.set_xscale("log"); a_.set_xlabel("parameters in the head")
            a_.set_title(ttl, fontweight="bold", loc="left", fontsize=10)
        if not np.isnan(LIN):
            ax[0].axhline(LIN, color=INK, lw=1.2, ls=":")
        ax[0].axhline(CEIL, color=MUTED, lw=1.2, ls="--")
        ax[0].legend(fontsize=7.5, ncol=2)
        fig.suptitle("Capacity against generalisation: circles = shallow bits, "
                     "squares = full encoding", fontweight="bold", fontsize=10.5,
                     x=0.01, ha="left")
        fig.tight_layout(rect=(0, 0, 1, 0.93))
        fig.savefig(os.path.join(root, "fig2_capacity.png"), dpi=150)
        plt.close(fig)

    # --- 3: training curves of the best run per head ---
    curves = []
    for h in HEADS:
        g = t[(t["head"] == h) & (t.bits == "shallow")]
        if not len(g):
            continue
        best = g.sort_values("val").iloc[-1]
        js = glob.glob(os.path.join(root, "shallow", best.label, "deephead_*.json"))
        if js:
            curves.append((h, best.label, json.load(open(js[0]))["history"]))
    if curves:
        fig, ax = plt.subplots(1, len(curves), figsize=(4.6 * len(curves), 3.8),
                               squeeze=False)
        for a_, (h, lab, hist) in zip(ax[0], curves):
            ep = [r["epoch"] for r in hist]
            a_.plot(ep, [r["train_auc"] for r in hist], color=HCOL[h], lw=1.6,
                    label="train")
            a_.plot(ep, [r["val_auc"] for r in hist], color=HCOL[h], lw=1.6,
                    ls="--", label="validation")
            if not np.isnan(LIN):
                a_.axhline(LIN, color=INK, lw=1.1, ls=":", label="linear head")
            a_.axhline(CEIL, color=MUTED, lw=1.1, ls="--", label="best tree")
            a_.set_title(f"{lab}", fontweight="bold", loc="left", fontsize=9.5)
            a_.set_xlabel("epoch"); a_.set_ylim(0.75, 1.02)
            a_.legend(fontsize=7.5)
        ax[0][0].set_ylabel("ranking score")
        fig.suptitle("Best run of each head: does the training curve run away from validation?",
                     fontweight="bold", fontsize=10, x=0.01, ha="left")
        fig.tight_layout(rect=(0, 0, 1, 0.92))
        fig.savefig(os.path.join(root, "fig3_curves.png"), dpi=150)
        plt.close(fig)
    print(f"[suite] figures -> {root}")


README = """DEEP HEADS ON A REGULARISED TREE ENCODING -- results bundle
=============================================================

Dataset: MagicTelescope (OpenML task 361065), two classes, 10 raw features.
Chosen because it is the binary dataset where the linear head fell clearly
short of the trees (0.9047 against a ceiling of 0.9377), so a deep head has
room to show a gain that is larger than the measurement noise.

The question
  The features are already regularised: deleting the deepest tree layers stops
  the model memorising. Does a deep network's extra capacity now turn into
  better predictions, or into a new way to overfit? And does the network need
  regularising of its own on top of the regularised features?

Held fixed everywhere
  forest 100 trees, depth 6, min 5 rows per leaf; out-of-bag honest bits;
  same split, seed 0, at most 16,000 rows; AdamW, learning rate 1e-3,
  batch size 256, up to 150 epochs, stopping when validation has not improved
  for 30 epochs, and the epoch with the best validation score is the one
  reported.

Heads
  linear      logistic regression, penalty chosen on validation. The reference.
  mlp         a plain multi-layer network on the bits
  tabresnet   the residual network used earlier in this project
  bittf       a transformer in which EVERY BIT is its own token, so every
              split attends to every other split. Each bit gets its own learned
              embedding, so the model learns what each individual split means
              and which splits matter together. This is the expensive option:
              attention cost grows with the square of the bit count, and the
              ordinary implementation would need about 17 GB for the shallow
              encoding at batch 256. It is made to fit with PyTorch's
              memory-efficient attention kernel, which never builds the full
              score matrix, plus a smaller batch (64) and mixed precision.

Sizes: small, medium, large. These are chosen so the knob moves the FIRST
layer, which on 1,454 bits holds 85-97% of every model's parameters and is
where the memorising happens.

Regularisation, six settings:
  none     nothing
  hidden   dropout 0.3 on the hidden activations only
  decay    AdamW weight decay 0.3 (0.01 was a no-op: 1.5% shrinkage per run)
  input    input dropout 0.3 -- drops whole BITS before the head sees them
  input5   input dropout 0.5
  all      hidden 0.3 + decay 0.3 + input 0.3
Input dropout is the in-network analogue of the bit-flip noise that worked on
the feature side, and the only setting that reaches the first layer.

Encodings
  shallow   depths 4 and 5 deleted, about 1,450 bits -- the recommended
            configuration from the feature-regularisation study
  full      all depths, about 4,700 bits

Files
  suite_long.csv       every run: head, size, regularisation, parameters, and
                       the ranking score, accuracy and log loss on train,
                       validation and test, plus the overfitting gap
  suite_summary.csv    the best run per head and encoding, against the linear
                       reference and the tree ceiling
  suite_results.xlsx   the same tables as an Excel workbook
  fig1_heads.png       test score and gap for every head, size and setting
  fig2_capacity.png    score and gap against parameter count, shallow vs full
  fig3_curves.png      training curves of the best run per head
  suite.log            the console log of the whole run
  <encoding>/<run>/run.log            that run's complete log
  <encoding>/<run>/deephead_*.json    that run's result and epoch history

How to read it
  gap = training ranking score minus validation ranking score, with training
  measured on clean bits. Near zero means no measurable overfitting.
  The ranking score has an uncertainty of about 0.01 on this test set, so
  differences smaller than that are ties.

Reproduce
  python run_deep_head_suite.py --part all --bundle
"""


def aggregate(root, bundle=False):
    t = load_rows(root)
    if t.empty:
        print("[suite] nothing to aggregate yet")
        return
    t = t.sort_values(["bits", "head", "size", "reg"])
    t.to_csv(os.path.join(root, "suite_long.csv"), index=False)
    CEIL = t.tree_ceiling.iloc[0]
    lin = t[(t["head"] == "linear") & (t.bits == "shallow")]
    LIN = float(lin.test.iloc[0]) if len(lin) else np.nan

    best = []
    for (bits, head), g in t.groupby(["bits", "head"]):
        r = g.sort_values("val").iloc[-1]        # chosen on validation, never test
        best.append(dict(encoding=bits, head=head, size=r["size"], reg=r.reg,
                         params=int(r.n_params), bits_used=int(r.n_bits),
                         train=round(r.train, 4), val=round(r.val, 4),
                         test=round(r.test, 4), gap=round(r.gap, 4),
                         test_acc=round(r.test_acc, 4),
                         test_logloss=round(r.test_logloss, 4),
                         vs_linear=round(r.test - LIN, 4) if not np.isnan(LIN) else None,
                         vs_tree=round(r.test - CEIL, 4), seconds=r.seconds))
    b = pd.DataFrame(best).sort_values(["encoding", "test"], ascending=[True, False])
    b.to_csv(os.path.join(root, "suite_summary.csv"), index=False)
    try:
        with pd.ExcelWriter(os.path.join(root, "suite_results.xlsx")) as xw:
            b.to_excel(xw, sheet_name="best_per_head", index=False)
            t.round(5).to_excel(xw, sheet_name="all_runs", index=False)
    except Exception as exc:  # noqa: BLE001
        print(f"[suite] workbook skipped ({exc}); the CSVs hold the same data")

    pd.set_option("display.width", 250)
    print("\n" + "=" * 98)
    print("BEST RUN PER HEAD  (chosen on validation, never on test)")
    print("=" * 98)
    print(b.to_string(index=False))
    print(f"\n  linear reference (shallow bits): {LIN:.4f}"
          if not np.isnan(LIN) else "\n  linear reference not run yet")
    print(f"  best tree on raw features:       {CEIL:.4f} "
          f"({t.best_tree.iloc[0].replace('_', ' ')})")
    print("  the ranking score is uncertain by about 0.01 here, so smaller "
          "differences are ties")
    with open(os.path.join(root, "README.txt"), "w") as f:
        f.write(README)
    figures(t, root)
    if bundle:
        import zipfile
        stamp = time.strftime("%Y%m%d")
        parent = os.path.dirname(os.path.abspath(root))
        path = os.path.join(parent, f"deep_head_{stamp}.zip")
        base = os.path.basename(os.path.abspath(root))
        n = 0
        with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
            for dirpath, dirnames, files in os.walk(root):
                # the cached forest and bits are a rebuildable convenience, and
                # large; they belong on Drive, not in the report bundle
                dirnames[:] = [d for d in dirnames if d != "cache"]
                for f in files:
                    full = os.path.join(dirpath, f)
                    z.write(full, os.path.join(base, os.path.relpath(full, root)))
                    n += 1
        print(f"[suite] bundle -> {path}  ({os.path.getsize(path) / 1e6:.1f} MB, "
              f"{n} files, cache excluded)")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="results/deep_head")
    ap.add_argument("--part", default="all", choices=["A", "B", "C", "all"])
    ap.add_argument("--ckpt-dir", default=None)
    ap.add_argument("--cache-dir", default=None)
    ap.add_argument("--sync-dir", default=None,
                    help="copy each finished run here (a Drive folder), so "
                         "an interrupted session loses at most one run")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--aggregate-only", action="store_true")
    ap.add_argument("--bundle", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.root, exist_ok=True)
    if not args.aggregate_only:
        run_all(args.root, plan(args.part), args.quick, args.ckpt_dir,
                args.cache_dir, args.sync_dir)
    aggregate(args.root, args.bundle)


if __name__ == "__main__":
    main()
