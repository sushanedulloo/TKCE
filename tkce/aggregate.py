"""Cross-dataset aggregation for the results table.

From a long results frame (one row per dataset x model x seed) we produce the
statistics an A* tabular paper reports:

  * per-dataset mean of the primary metric (averaged over seeds)
  * per-dataset rank (1 = best) and the average rank per model
  * Wilcoxon signed-rank tests: each model vs a reference (e.g. best tree)
  * a critical-difference (Nemenyi) diagram over average ranks (Demsar 2006)

Higher-is-better metrics only (AUC, R^2); ranks are computed accordingly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def summarize(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Mean +/- std of `metric` per (dataset, model) across seeds."""
    g = df.groupby(["dataset", "model"])[metric]
    out = g.agg(["mean", "std", "count"]).reset_index()
    return out


def rank_table(df: pd.DataFrame, metric: str):
    """Return (pivot of mean metric, per-dataset ranks, avg-rank series)."""
    summ = summarize(df, metric)
    pivot = summ.pivot(index="dataset", columns="model", values="mean")
    # Rank per dataset: higher metric -> better -> rank 1.
    ranks = pivot.rank(axis=1, ascending=False, method="average")
    avg_rank = ranks.mean(axis=0).sort_values()
    return pivot, ranks, avg_rank


def wilcoxon_vs_reference(pivot: pd.DataFrame, reference: str) -> pd.DataFrame:
    """Wilcoxon signed-rank of each model vs `reference` across datasets."""
    from scipy.stats import wilcoxon
    rows = []
    ref = pivot[reference]
    for m in pivot.columns:
        if m == reference:
            continue
        a, b = pivot[m].dropna(), ref.loc[pivot[m].dropna().index]
        common = a.index.intersection(b.index)
        a, b = pivot.loc[common, m], pivot.loc[common, reference]
        try:
            stat, p = wilcoxon(a, b)
        except Exception:  # noqa: BLE001 - all-equal or too few pairs
            stat, p = float("nan"), float("nan")
        rows.append({"model": m, "vs": reference, "mean_diff": float((a - b).mean()),
                     "wins": int((a > b).sum()), "losses": int((a < b).sum()),
                     "p_value": float(p)})
    return pd.DataFrame(rows).sort_values("mean_diff", ascending=False)


def nemenyi_cd(avg_rank: pd.Series, n_datasets: int, alpha: str = "0.05") -> float:
    """Critical difference for the Nemenyi test."""
    k = len(avg_rank)
    # Studentized-range-based q_alpha critical values (alpha=0.05), infinite df.
    q05 = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949,
           8: 3.031, 9: 3.102, 10: 3.164, 11: 3.219, 12: 3.268, 13: 3.313,
           14: 3.354, 15: 3.391, 16: 3.426}
    q = q05.get(k, 3.426 + 0.03 * (k - 16))
    return float(q * np.sqrt(k * (k + 1) / (6.0 * n_datasets)))


def critical_difference_diagram(avg_rank: pd.Series, cd: float, save_path: str,
                                title: str = "Critical-difference diagram"):
    """Draw a Demsar-style CD diagram of average ranks."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(avg_rank.index)
    ranks = avg_rank.values
    order = np.argsort(ranks)
    names = [names[i] for i in order]
    ranks = ranks[order]
    k = len(names)

    lo, hi = 1, int(np.ceil(ranks.max()))
    fig, ax = plt.subplots(figsize=(9, 0.5 * k + 1.6))
    ax.set_xlim(lo - 0.5, hi + 0.5)
    ax.set_ylim(0, k + 1)
    ax.invert_yaxis()
    ax.axis("off")

    # Rank axis.
    ax.plot([lo, hi], [0.4, 0.4], "k-", lw=1)
    for x in range(lo, hi + 1):
        ax.plot([x, x], [0.35, 0.45], "k-", lw=1)
        ax.text(x, 0.15, str(x), ha="center", va="bottom", fontsize=8)

    for i, (nm, r) in enumerate(zip(names, ranks)):
        y = i + 1
        ax.plot([r, r], [0.45, y], "k-", lw=0.8)
        ax.plot([r, lo - 0.4], [y, y], "k-", lw=0.8)
        ax.text(lo - 0.5, y, f"{nm}  ({r:.2f})", ha="right", va="center", fontsize=8)

    # CD bar.
    ax.plot([lo, lo + cd], [k + 0.5, k + 0.5], "r-", lw=2.5)
    ax.text(lo + cd / 2, k + 0.75, f"CD = {cd:.2f}", ha="center", va="top",
            fontsize=8, color="red")
    ax.set_title(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def full_report(df: pd.DataFrame, metric: str, out_dir: str, reference: str):
    """Write pivot, ranks, wilcoxon, and a CD diagram. Returns a dict of frames."""
    import os
    os.makedirs(out_dir, exist_ok=True)
    pivot, ranks, avg_rank = rank_table(df, metric)
    wil = wilcoxon_vs_reference(pivot, reference) if reference in pivot else None
    cd = nemenyi_cd(avg_rank, n_datasets=pivot.shape[0])

    pivot.to_csv(os.path.join(out_dir, "pivot_mean.csv"))
    ranks.to_csv(os.path.join(out_dir, "ranks.csv"))
    avg_rank.to_frame("avg_rank").to_csv(os.path.join(out_dir, "avg_rank.csv"))
    if wil is not None:
        wil.to_csv(os.path.join(out_dir, "wilcoxon.csv"), index=False)
    try:
        critical_difference_diagram(
            avg_rank, cd, os.path.join(out_dir, "cd_diagram.png"),
            title=f"CD diagram ({metric}, {pivot.shape[0]} datasets)")
    except Exception as e:  # noqa: BLE001
        print(f"[aggregate] CD diagram skipped: {e}")
    return {"pivot": pivot, "ranks": ranks, "avg_rank": avg_rank,
            "wilcoxon": wil, "cd": cd}
