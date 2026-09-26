"""Regenerate the three figures for the feature-regularisation report.

Reads data.json (the measured results, transcribed from the run output and
reconciled against the deltas the runs printed themselves) and writes the PDF
figures the LaTeX source includes. Run from this directory:

    python make_figures.py && tectonic -X compile report.tex --outdir .
"""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

D = json.load(open("data.json"))
ROWS, CEIL = D["rows"], D["ceil"]
plt.rcParams.update({
    "font.size": 9.5, "axes.grid": True, "grid.alpha": .22, "grid.linewidth": .5,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#9a9a96", "figure.facecolor": "white",
    "axes.facecolor": "white", "legend.frameon": False, "font.family": "DejaVu Sans"})
# Validated categorical palette; colour carries identity and every series is labelled.
BLUE, ORANGE, AQUA, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7"
INK, MUTED = "#0b0b0b", "#52514e"
FAM = {"baseline": INK, "uniform": ORANGE, "ramp": AQUA,
       "delete": BLUE, "delete+ramp": VIOLET}


def family(name):
    """Group an arm by the kind of intervention it applies."""
    if name.startswith("00"):
        return "baseline"
    if name.startswith(("10", "21")):
        return "uniform"
    if name.startswith("20"):
        return "ramp"
    if name in ("30_drop_L5", "31_drop_L45", "40_keep_L012"):
        return "delete"
    return "delete+ramp"


# The three stability-repeat arms duplicate configurations already listed, so
# they are excluded from the figures and used only for the spread statement.
MAIN = [r for r in ROWS if not r[0].startswith("5")]
SHORT = {
    "00_baseline": "baseline (all 5,401 bits)", "10_unif_0.05": "uniform p=0.05",
    "10_unif_0.1": "uniform p=0.10", "10_unif_0.2": "uniform p=0.20",
    "10_unif_0.4": "uniform p=0.40", "20_ramp_0.1": "ramp P=0.1",
    "20_ramp_0.2": "ramp P=0.2", "20_ramp_0.5": "ramp P=0.5",
    "21_ctrl_0.08": "uniform p=0.08 (control for ramp 0.1)",
    "21_ctrl_0.16": "uniform p=0.16 (control for ramp 0.2)",
    "21_ctrl_0.4": "uniform p=0.40 (control for ramp 0.5)",
    "30_drop_L5": "drop depth 5", "31_drop_L45": "drop depths 4,5",
    "32_drop_L5_ramp02": "drop 5 + ramp 0.2",
    "33_drop_L45_ramp02": "drop 4,5 + ramp 0.2",
    "34_drop_L45_ramp02rel": "drop 4,5 + ramp 0.2 (rescaled)",
    "40_keep_L012": "keep depths 0-2 only", "41_keep_L012_ramp": "keep 0-2 + ramp 0.2"}


def fig_tradeoff():
    """Overfitting gap against test AUC — the headline figure."""
    fig, a = plt.subplots(figsize=(7.4, 4.6))
    a.axhline(CEIL, ls="--", color=MUTED, lw=1.2)
    a.text(0.004, CEIL + 0.00035, f"LightGBM ceiling {CEIL:.4f}", fontsize=8, color=MUTED)
    for name, bits, _kept, _mp, _tr, _va, te, gap, _sd, _f in MAIN:
        a.scatter(gap, te, s=30 + bits / 60, color=FAM[family(name)], alpha=.9,
                  zorder=3, edgecolor="white", linewidth=.9)
    b = MAIN[0]
    a.annotate("baseline\nall 5,401 bits", (b[7], b[6]), xytext=(-10, -30),
               textcoords="offset points", fontsize=8, color=INK, ha="center",
               arrowprops=dict(arrowstyle="->", color=MUTED, lw=.8))
    k = [r for r in MAIN if r[0] == "40_keep_L012"][0]
    a.annotate("keep depths 0-2 only:\n700 bits, gap 4x smaller,\ntest AUC unchanged",
               (k[7], k[6]), xytext=(6, 22), textcoords="offset points", fontsize=8,
               color=INK, arrowprops=dict(arrowstyle="->", color=MUTED, lw=.8))
    w = [r for r in MAIN if r[0] == "20_ramp_0.5"][0]
    a.annotate("heaviest noise: the only\narms that drop below the trees",
               (w[7], w[6]), xytext=(16, -2), textcoords="offset points", fontsize=8,
               color=INK, va="center", arrowprops=dict(arrowstyle="->", color=MUTED, lw=.8))
    a.set_xlabel("overfitting  (train AUC $-$ validation AUC, measured on clean bits)")
    a.set_ylabel("test AUC")
    a.set_xlim(-0.004, 0.108)
    a.set_ylim(0.8412, 0.8474)
    a.set_title("Overfitting falls by more than ten times at no cost in test AUC",
                fontweight="bold", loc="left", fontsize=10.5)
    a.legend(handles=[Patch(color=c, label=l) for l, c in FAM.items()], fontsize=8,
             loc="lower right", ncol=2,
             title="marker area $\\propto$ number of bits", title_fontsize=7.5)
    fig.tight_layout()
    fig.savefig("fig1_tradeoff.pdf")


def fig_bars():
    """Test AUC and the overfitting gap, one bar per arm."""
    order = sorted(MAIN, key=lambda r: (
        ["baseline", "uniform", "ramp", "delete", "delete+ramp"].index(family(r[0])),
        r[0]))
    labels = [SHORT[r[0]] for r in order]
    cols = [FAM[family(r[0])] for r in order]
    fig, ax = plt.subplots(1, 2, figsize=(9.6, 5.6), gridspec_kw={"wspace": 0.62})
    for axis, idx, title, xlabel in [
            (ax[0], 6, "Test AUC", "test AUC"),
            (ax[1], 7, "Overfitting gap", "train $-$ validation AUC")]:
        vals = [r[idx] for r in order]
        axis.barh(range(len(order)), vals, color=cols, edgecolor="white",
                  lw=1.0, height=.72)
        for i, x in enumerate(vals):
            axis.text(x, i, f" {x:.4f}", va="center", fontsize=7.5, color=INK)
        axis.set_yticks(range(len(order)))
        axis.set_yticklabels(labels, fontsize=7.5)
        axis.invert_yaxis()
        axis.grid(alpha=.25, axis="x")
        axis.set_xlabel(xlabel)
        axis.set_title(title, fontweight="bold", loc="left", fontsize=10)
    ax[0].axvline(CEIL, ls="--", color=MUTED, lw=1.2)
    ax[0].set_xlim(0.838, 0.8492)
    ax[0].text(CEIL, -1.0, f"LightGBM {CEIL:.4f}", fontsize=7, color=MUTED, ha="center")
    ax[1].set_xlim(0, 0.112)
    fig.savefig("fig2_bars.pdf", bbox_inches="tight")


def fig_profile():
    """Left: flip probability by depth. Right: each ramp beside its matched control."""
    import numpy as np
    fig, ax = plt.subplots(1, 2, figsize=(9.6, 3.9))
    a = ax[0]
    # (label, p at depths 0..5, colour, linestyle) — read off the run headers.
    profiles = {
        "uniform p=0.20": ([.2] * 6, ORANGE, "-"),
        "ramp P=0.2 (all layers)": ([0, .04, .08, .12, .16, .20], AQUA, "-"),
        "ramp P=0.5 (all layers)": ([0, .1, .2, .3, .4, .5], AQUA, "--"),
        "drop 4,5 + ramp 0.2": ([0, .04, .08, .12, 0, 0], VIOLET, "-"),
        "drop 4,5 + ramp 0.2 rescaled": ([0, .0667, .1333, .2, 0, 0], VIOLET, "--"),
        "keep 0-2 + ramp 0.2": ([0, .1, .2, 0, 0, 0], BLUE, "-")}
    for label, (ys, colour, ls) in profiles.items():
        a.plot(range(6), ys, ls, marker="o", ms=4, lw=1.5, color=colour,
               label=label, alpha=.9)
    a.set_xlabel("tree depth of the split")
    a.set_ylabel("flip probability")
    a.set_title("What each arm does to each layer\n(zero means deleted or untouched)",
                fontweight="bold", loc="left", fontsize=9.5)
    a.legend(fontsize=7, loc="upper left")
    a.set_ylim(-0.02, 0.56)

    b = ax[1]
    # (label, ramp test AUC, matched-uniform test AUC) at equal average flip rate.
    pairs = [("ramp P=0.1\nvs uniform 0.08", 0.8457, 0.8456),
             ("ramp P=0.2\nvs uniform 0.16", 0.8442, 0.8460),
             ("ramp P=0.5\nvs uniform 0.40", 0.8424, 0.8433)]
    x = np.arange(len(pairs))
    w = 0.34
    b.bar(x - w / 2, [p[1] for p in pairs], w, color=AQUA, edgecolor="white",
          lw=1, label="depth ramp")
    b.bar(x + w / 2, [p[2] for p in pairs], w, color=ORANGE, edgecolor="white",
          lw=1, label="uniform, same budget")
    for i, p in enumerate(pairs):
        b.text(i - w / 2, p[1] + 0.0002, f"{p[1]:.4f}", ha="center", fontsize=7)
        b.text(i + w / 2, p[2] + 0.0002, f"{p[2]:.4f}", ha="center", fontsize=7)
    b.axhline(CEIL, ls="--", color=MUTED, lw=1.2)
    b.text(2.42, CEIL + 0.00025, "LightGBM", fontsize=7, color=MUTED, ha="right")
    b.set_xticks(x)
    b.set_xticklabels([p[0] for p in pairs], fontsize=7.5)
    b.set_ylim(0.8405, 0.8475)
    b.set_ylabel("test AUC")
    b.set_title("Equal noise budget: the depth shape gives no gain",
                fontweight="bold", loc="left", fontsize=9.5)
    b.legend(fontsize=7.5, loc="lower left")
    fig.tight_layout()
    fig.savefig("fig3_profile.pdf")


if __name__ == "__main__":
    fig_tradeoff()
    fig_bars()
    fig_profile()
    print("wrote fig1_tradeoff.pdf, fig2_bars.pdf, fig3_profile.pdf")
