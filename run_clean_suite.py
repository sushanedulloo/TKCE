"""Three-view ablation (x / x+tree / x+tree+deep) across 5 CLEAN datasets.

Datasets are from the Grinsztajn numeric-classification suite (OpenML suite 337),
chosen to EXCLUDE everything on TabReD's leak list (arXiv:2406.19380) — no
eye_movements, no electricity:

    361062  pol
    361063  house_16H
    361065  MagicTelescope
    361277  california
    361055  credit

For each dataset this driver invokes run_fusion.py once per encoding with
--views 'x;x+tree;x+tree+deep' (so all three configs share the same split,
the same encoding RF, and the same tree ceiling), then aggregates every JSON
into one cross-dataset summary table + CSV.

Colab (GPU):
    python -u run_clean_suite.py --device auto
    # both encodings (the paper's naive-vs-honest comparison; ~2x the time):
    python -u run_clean_suite.py --encodings infold,oob --device auto
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys

DATASETS = {  # task_id -> short name (all clean per TabReD)
    361062: "pol",
    361063: "house_16H",
    361065: "MagicTelescope",
    361277: "california",
    361055: "credit",
}
VIEWS = "x;x+tree;x+tree+deep"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default=",".join(str(t) for t in DATASETS),
                    help="comma-separated OpenML task ids (default: the 5 clean ones)")
    ap.add_argument("--encodings", default="oob",
                    help="comma list from {infold,oob}; 'infold,oob' runs both")
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--ensemble", type=int, default=2,
                    help="members per view config (2 keeps the 5-dataset suite ~2h)")
    ap.add_argument("--rf-trees", type=int, default=100)
    ap.add_argument("--rf-depth", type=int, default=6)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--l1", type=float, default=5e-5)
    ap.add_argument("--weight-decay", type=float, default=1e-3)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", default="results/fusion/clean")
    args = ap.parse_args()

    tasks = [int(t) for t in args.tasks.split(",")]
    encodings = [e.strip() for e in args.encodings.split(",") if e.strip()]
    os.makedirs(args.out, exist_ok=True)

    n_runs = len(tasks) * len(encodings)
    print(f"[suite] {len(tasks)} datasets x {len(encodings)} encoding(s) "
          f"x 3 views x {args.ensemble} members "
          f"= {n_runs * 3 * args.ensemble} trainings of {args.epochs} epochs\n",
          flush=True)

    done, failed = [], []
    for tid in tasks:
        name = DATASETS.get(tid, f"task{tid}")
        for enc in encodings:
            outdir = os.path.join(args.out, f"{name}_{enc}")
            print(f"\n{'#' * 26}  {name} ({tid}) | encoding={enc}  {'#' * 26}",
                  flush=True)
            cmd = [sys.executable, "-u", "run_fusion.py",
                   "--task", str(tid), "--views", VIEWS,
                   "--encoding", enc, "--ensemble", str(args.ensemble),
                   "--epochs", str(args.epochs),
                   "--rf-trees", str(args.rf_trees), "--rf-depth", str(args.rf_depth),
                   "--dropout", str(args.dropout), "--l1", str(args.l1),
                   "--weight-decay", str(args.weight_decay), "--lr", str(args.lr),
                   "--batch-size", str(args.batch_size),
                   "--device", args.device, "--out", outdir]
            r = subprocess.run(cmd)
            (done if r.returncode == 0 else failed).append((name, enc))
            if r.returncode != 0:
                print(f"[suite] !! {name}/{enc} FAILED (exit {r.returncode}) — continuing",
                      flush=True)

    # ---------------- aggregate ----------------
    rows = []
    for name, enc in done:
        js = glob.glob(os.path.join(args.out, f"{name}_{enc}", "fusion_*.json"))
        if not js:
            continue
        s = json.load(open(js[0]))
        for r in s["results"]:
            rows.append(dict(dataset=s["dataset"], encoding=enc, views=r["views"],
                             test_auc=r["test_auc"], ceiling=s["tree_ceiling"],
                             vs_ceiling=round(r["test_auc"] - s["tree_ceiling"], 4)))

    csv_path = os.path.join(args.out, "summary.csv")
    with open(csv_path, "w") as f:
        f.write("dataset,encoding,views,test_auc,tree_ceiling,vs_ceiling\n")
        for r in rows:
            f.write(f"{r['dataset']},{r['encoding']},{r['views']},"
                    f"{r['test_auc']:.4f},{r['ceiling']:.4f},{r['vs_ceiling']:+.4f}\n")

    print("\n" + "=" * 78)
    print("CROSS-DATASET SUMMARY (test AUC; delta = vs tree ceiling)")
    print("=" * 78)
    print(f"{'dataset':16s} {'enc':7s} {'ceiling':>8s} {'x':>8s} {'x+tree':>8s} "
          f"{'FULL':>8s}   best view")
    key = {}
    for r in rows:
        key.setdefault((r["dataset"], r["encoding"]),
                       {"ceiling": r["ceiling"]})[r["views"]] = r["test_auc"]
    for (d, e), v in key.items():
        vx, vt, vf = v.get("x"), v.get("x+tree"), v.get("x+tree+deep")
        opts = {n: a for n, a in
                (("x", vx), ("x+tree", vt), ("FULL", vf)) if a is not None}
        best = max(opts, key=opts.get) if opts else "-"
        fmt = lambda a: f"{a:8.4f}" if a is not None else "       -"
        print(f"{d:16s} {e:7s} {v['ceiling']:8.4f} {fmt(vx)} {fmt(vt)} {fmt(vf)}   "
              f"{best}{' (beats ceiling!)' if opts and opts[best] > v['ceiling'] else ''}")
    if failed:
        print(f"\n[suite] FAILED runs: {failed}")
    print(f"\n[suite] summary CSV -> {csv_path}")
    print(f"[suite] per-dataset figures/JSONs under {args.out}/<dataset>_<encoding>/")


if __name__ == "__main__":
    main()
