"""Build every figure and every LaTeX table for the suite report FROM THE CSVs.

Nothing here is typed by hand: tables.tex and the figures are generated from
suite_long.csv / suite_verdict.csv, so the report cannot drift from the data.
Vocabulary is plain: "ranking score" for the classification metric,
"explained-variance score" for regression, no acronyms anywhere in labels.
"""
import os, sys
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize

ROOT = sys.argv[1]
OUT = os.path.dirname(os.path.abspath(__file__))
t = pd.read_csv(os.path.join(ROOT, "suite_long.csv"))
v = pd.read_csv(os.path.join(ROOT, "suite_verdict.csv"))

ORDER = ["credit", "house_16H", "MagicTelescope", "pendigits", "covertype", "cpu_act", "wine_quality"]
KIND = {"credit": "two-class", "house_16H": "two-class", "MagicTelescope": "two-class",
        "pendigits": "ten-class", "covertype": "seven-class", "cpu_act": "regression",
        "wine_quality": "regression"}
ARMS = ["00_baseline", "10_unif_0.2", "20_ramp_0.2", "21_ctrl_0.16", "30_drop_L5",
        "31_drop_L45", "33_drop_L45_ramp02", "40_keep_L012"]
NAME = {"00_baseline": "all bits, no change",
        "10_unif_0.2": "flip every bit, p = 0.20",
        "20_ramp_0.2": "flips rising with depth, P = 0.2",
        "21_ctrl_0.16": "flip every bit, p = 0.16 (same budget as the ramp)",
        "30_drop_L5": "delete depth 5",
        "31_drop_L45": "delete depths 4 and 5",
        "33_drop_L45_ramp02": "delete 4 and 5, then ramp 0.2",
        "40_keep_L012": "keep only depths 0, 1, 2"}
SHORT = {"00_baseline": "baseline", "10_unif_0.2": "flip 0.20", "20_ramp_0.2": "ramp 0.2",
         "21_ctrl_0.16": "flip 0.16", "30_drop_L5": "delete 5", "31_drop_L45": "delete 4,5",
         "33_drop_L45_ramp02": "delete 4,5\n+ ramp", "40_keep_L012": "keep 0-2"}
BLUE, ORANGE, AQUA, VIOLET, AMBER = "#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7", "#b5730f"
INK, MUTED, FAINT = "#0b0b0b", "#52514e", "#c8c8c4"
DCOL = {"credit": BLUE, "house_16H": ORANGE, "MagicTelescope": AQUA, "pendigits": VIOLET,
        "covertype": AMBER, "cpu_act": "#e34948", "wine_quality": "#008300"}
plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": .22, "grid.linewidth": .5,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#9a9a96", "figure.facecolor": "white",
                     "legend.frameon": False, "font.family": "DejaVu Sans"})

tree = t[t.view == "tree"].copy()
raw = t[t.view == "x"].set_index("dataset")
base = tree[tree.arm == "00_baseline"].set_index("dataset")
keep = tree[tree.arm == "40_keep_L012"].set_index("dataset")


def metric_word(ds):
    return "explained-variance score" if base.loc[ds, "task_type"] == "regression" else "ranking score"


# ============================================================ FIGURE 1
# Where the linear model sits between "raw features only" and "best tree".
fig, a = plt.subplots(figsize=(8.4, 4.4))
for i, ds in enumerate(ORDER):
    lo, hi = raw.loc[ds, "test"], base.loc[ds, "tree_ceiling"]
    span = hi - lo
    a.plot([0, 1], [i, i], color=FAINT, lw=6, solid_capstyle="round", zorder=1)
    pb = (base.loc[ds, "test"] - lo) / span
    pk = (keep.loc[ds, "test"] - lo) / span
    a.scatter([pb], [i], s=110, color=DCOL[ds], zorder=3, edgecolor="white", lw=1.2)
    a.scatter([pk], [i], s=110, color=DCOL[ds], marker="D", zorder=3, edgecolor="white", lw=1.2)
    a.text(1.03, i, f"raw {lo:.3f}  |  all bits {base.loc[ds,'test']:.4f}  |  "
                    f"keep 0-2 {keep.loc[ds,'test']:.4f}  |  tree {hi:.4f}",
           va="center", fontsize=7.6, color=MUTED)
a.axvline(0, color=MUTED, lw=1); a.axvline(1, color=MUTED, lw=1, ls="--")
a.text(0, -0.9, "raw features only\n(no tree bits)", ha="center", fontsize=8, color=MUTED)
a.text(1, -0.9, "best tree model", ha="center", fontsize=8, color=MUTED)
a.set_yticks(range(len(ORDER)))
a.set_yticklabels([f"{d}\n({KIND[d]})" for d in ORDER], fontsize=8.5)
a.set_xlim(-0.12, 1.12); a.set_ylim(len(ORDER) - 0.4, -1.5); a.invert_yaxis()
a.set_ylim(-1.6, len(ORDER) - 0.4); a.invert_yaxis()
a.set_xlabel("share of the distance from 'raw features' to 'best tree' that the linear model covers")
a.scatter([], [], s=90, color=MUTED, label="all bits (5,400 or so)")
a.scatter([], [], s=90, color=MUTED, marker="D", label="keep only depths 0, 1, 2 (about 700 bits)")
a.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2, fontsize=8)
a.set_title("Only on credit does the linear model on tree bits reach the best tree",
            fontweight="bold", loc="left", fontsize=10.5)
a.grid(False)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig1_position.pdf"), bbox_inches="tight"); fig.savefig(os.path.join(OUT, "fig1_position.png"), dpi=130, bbox_inches="tight")
plt.close(fig)

# ============================================================ FIGURE 2
# Train came down toward test: dumbbells for baseline and keep 0-2.
fig, ax = plt.subplots(1, 2, figsize=(11, 4.6), gridspec_kw={"width_ratios": [1.35, 1]})
a = ax[0]
y = 0
ticks, labels = [], []
for ds in ORDER:
    for arm, mk, lab in [("00_baseline", "o", "all bits"), ("40_keep_L012", "D", "keep 0-2")]:
        r = tree[(tree.dataset == ds) & (tree.arm == arm)].iloc[0]
        a.plot([r.test, r.train], [y, y], color=DCOL[ds], lw=2.2, alpha=.85)
        a.scatter([r.test], [y], s=55, color="white", edgecolor=DCOL[ds], lw=1.8, zorder=3)
        a.scatter([r.train], [y], s=55, color=DCOL[ds], marker=mk, zorder=3, edgecolor="white")
        a.text(max(r.test, r.train) + 0.012, y, f"gap {r.gap:+.4f}", va="center", fontsize=7.4, color=MUTED)
        ticks.append(y); labels.append(f"{ds}  ·  {lab}")
        y += 1
    y += 0.5
a.set_yticks(ticks); a.set_yticklabels(labels, fontsize=7.6); a.invert_yaxis()
a.set_xlabel("score on the primary metric  (hollow = test,  filled = training on clean bits)")
a.set_xlim(0.2, 1.08)
a.set_title("Training score falls toward the test score\nwhen the deep bits are removed",
            fontweight="bold", loc="left", fontsize=10)
b = ax[1]
cls = [d for d in ORDER if base.loc[d, "task_type"] == "classification"]
x = np.arange(len(cls)); w = 0.2
for k, (arm, col, lab) in enumerate([("00_baseline", BLUE, "all bits: training"),
                                      ("40_keep_L012", AQUA, "keep 0-2: training")]):
    tr = [tree[(tree.dataset == d) & (tree.arm == arm)].train_accuracy.iloc[0] for d in cls]
    te = [tree[(tree.dataset == d) & (tree.arm == arm)].test_accuracy.iloc[0] for d in cls]
    b.bar(x + (2 * k - 1.5) * w, tr, w, color=col, edgecolor="white", label=lab)
    b.bar(x + (2 * k - 0.5) * w, te, w, color=col, alpha=.4, edgecolor="white",
          label=lab.replace("training", "test"))
    for i in range(len(cls)):
        b.text(x[i] + (2 * k - 1.5) * w, tr[i] + 0.006, f"{tr[i]:.3f}", ha="center", fontsize=6.3, rotation=90)
        b.text(x[i] + (2 * k - 0.5) * w, te[i] + 0.006, f"{te[i]:.3f}", ha="center", fontsize=6.3, rotation=90)
b.set_xticks(x); b.set_xticklabels(cls, fontsize=8, rotation=15)
b.set_ylim(0.6, 1.08); b.set_ylabel("accuracy")
b.set_title("Accuracy on the classification datasets:\ntraining accuracy comes off the ceiling",
            fontweight="bold", loc="left", fontsize=9.5)
b.legend(fontsize=7.2, ncol=2, loc="upper left")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig2_train_down.pdf"), bbox_inches="tight"); fig.savefig(os.path.join(OUT, "fig2_train_down.png"), dpi=130, bbox_inches="tight")
plt.close(fig)

# ============================================================ FIGURE 3
# Heat maps: change in test vs baseline, and gap ratio, datasets x arms.
arms = ARMS[1:]
dT = np.zeros((len(ORDER), len(arms))); gR = np.zeros_like(dT)
for i, ds in enumerate(ORDER):
    b_ = base.loc[ds]
    for j, arm in enumerate(arms):
        r = tree[(tree.dataset == ds) & (tree.arm == arm)].iloc[0]
        dT[i, j] = r.test - b_.test
        gR[i, j] = r.gap / b_.gap if b_.gap >= 0.01 else np.nan
div = LinearSegmentedColormap.from_list("d", [ORANGE, "#ffffff", BLUE])
seq = LinearSegmentedColormap.from_list("s", ["#ffffff", BLUE])
fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.6))
for a_, M, cmap, norm, ttl, fmt in [
        (ax[0], dT, div, Normalize(-0.02, 0.02),
         "Change in TEST score versus 'all bits'\n(blue = better, orange = worse; band of +/-0.01 is 'no real change')", "{:+.4f}"),
        (ax[1], gR, seq, Normalize(0, 1.5),
         "Overfitting gap as a fraction of the 'all bits' gap\n(lighter = less overfitting; blank = baseline gap under 0.01, nothing to reduce)", "{:.2f}")]:
    a_.imshow(np.clip(M, norm.vmin, norm.vmax), cmap=cmap, norm=norm, aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            val = M[i, j]
            if np.isnan(val):
                a_.text(j, i, "–", ha="center", va="center", fontsize=8, color=MUTED); continue
            txt = fmt.format(val) if abs(val) < 10 else f"{val:+.1f}"
            a_.text(j, i, txt, ha="center", va="center", fontsize=7.6,
                    color="white" if (cmap is seq and val > 1.0) or (cmap is div and abs(val) > 0.014) else INK)
    a_.set_xticks(range(len(arms))); a_.set_xticklabels([SHORT[x] for x in arms], fontsize=8)
    a_.set_yticks(range(len(ORDER))); a_.set_yticklabels([f"{d} ({KIND[d]})" for d in ORDER], fontsize=8)
    a_.set_title(ttl, fontweight="bold", loc="left", fontsize=9.5); a_.grid(False)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig3_heatmaps.pdf"), bbox_inches="tight"); fig.savefig(os.path.join(OUT, "fig3_heatmaps.png"), dpi=130, bbox_inches="tight")
plt.close(fig)

# ============================================================ FIGURE 4
# Calibration: test log loss for baseline / keep 0-2 / flip 0.20 / ramp 0.2 (classification).
fig, a = plt.subplots(figsize=(8.6, 4))
x = np.arange(len(cls)); w = 0.2
for k, (arm, col) in enumerate([("00_baseline", MUTED), ("40_keep_L012", BLUE),
                                ("20_ramp_0.2", AQUA), ("10_unif_0.2", ORANGE)]):
    vals = [tree[(tree.dataset == d) & (tree.arm == arm)].test_logloss.iloc[0] for d in cls]
    a.bar(x + (k - 1.5) * w, vals, w, color=col, edgecolor="white", label=NAME[arm].split(" (")[0])
    for i, vv in enumerate(vals):
        a.text(x[i] + (k - 1.5) * w, vv + 0.01, f"{vv:.3f}", ha="center", fontsize=6.4, rotation=90)
a.set_xticks(x); a.set_xticklabels(cls, fontsize=8.5)
a.set_ylabel("log loss on the test split  (lower = better calibrated)")
a.set_ylim(0, max(tree[tree.dataset.isin(cls)].test_logloss) * 1.35)
a.set_title("Noise damages calibration; deleting deep bits does not",
            fontweight="bold", loc="left", fontsize=10.5)
a.legend(fontsize=8, ncol=2)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig4_calibration.pdf"), bbox_inches="tight"); fig.savefig(os.path.join(OUT, "fig4_calibration.png"), dpi=130, bbox_inches="tight")
plt.close(fig)

# ============================================================ FIGURE 5
# Ramp vs matched-budget flat control, classification and regression panels.
fig, ax = plt.subplots(1, 2, figsize=(11, 4), gridspec_kw={"width_ratios": [1.5, 1]})
for a_, subset, ttl in [(ax[0], cls, "Classification: ranking score"),
                        (ax[1], [d for d in ORDER if d not in cls], "Regression: explained-variance score")]:
    x = np.arange(len(subset)); w = 0.36
    rv = [tree[(tree.dataset == d) & (tree.arm == "20_ramp_0.2")].test.iloc[0] for d in subset]
    cv = [tree[(tree.dataset == d) & (tree.arm == "21_ctrl_0.16")].test.iloc[0] for d in subset]
    bv = [base.loc[d, "test"] for d in subset]
    a_.bar(x - w / 2, rv, w, color=AQUA, edgecolor="white", label="flips rising with depth, P = 0.2")
    a_.bar(x + w / 2, cv, w, color=ORANGE, edgecolor="white", label="flip every bit, p = 0.16 (same budget)")
    for i, d in enumerate(subset):
        a_.plot([x[i] - w, x[i] + w], [bv[i], bv[i]], color=INK, lw=1.4, ls=":")
        a_.text(x[i] - w / 2, rv[i] + (0.002 if rv[i] > 0 else -0.08), f"{rv[i]:.3f}", ha="center", fontsize=7)
        a_.text(x[i] + w / 2, cv[i] + (0.002 if cv[i] > 0 else -0.08), f"{cv[i]:.3f}", ha="center", fontsize=7)
    a_.plot([], [], color=INK, lw=1.4, ls=":", label="all bits, no noise")
    a_.set_xticks(x); a_.set_xticklabels(subset, fontsize=8.5)
    a_.set_title(ttl, fontweight="bold", loc="left", fontsize=10)
ax[0].set_ylim(0.82, 1.03); ax[0].legend(fontsize=7.6, loc="upper left")
ax[1].set_ylim(-2.4, 1.15); ax[1].axhline(0, color=MUTED, lw=.8)
ax[1].text(-0.45, -1.35, "0 = no better than guessing the average", fontsize=7, color=MUTED, ha="left")
fig.suptitle("Depth-shaped noise versus flat noise at the same total amount",
             fontweight="bold", fontsize=10.5, x=0.01, ha="left")
fig.tight_layout(rect=(0, 0, 1, 0.94)); fig.savefig(os.path.join(OUT, "fig5_ramp_vs_flat.pdf"), bbox_inches="tight"); fig.savefig(os.path.join(OUT, "fig5_ramp_vs_flat.png"), dpi=130, bbox_inches="tight")
plt.close(fig)

# ============================================================ TABLES (LaTeX)
def f4(x): return f"{x:.4f}"
def tex(name): return name.replace("_", "\\_")
def sgn(x): return f"{x:+.4f}"
L = []

# --- overview / verdict table ---
L.append("% ---- overview table (generated) ----")
L.append("\\newcommand{\\taboverview}{%")
L.append("\\begin{tabular}{llrrrrrrrrrc}\n\\toprule")
L.append("\\textbf{Dataset} & \\textbf{Kind} & \\textbf{Best tree} & \\textbf{Raw only} & "
         "\\multicolumn{3}{c}{\\textbf{All bits}} & \\multicolumn{3}{c}{\\textbf{Keep depths 0--2}} & "
         "\\textbf{Test change} & \\textbf{Holds}\\\\")
L.append("\\cmidrule(lr){5-7}\\cmidrule(lr){8-10}")
L.append(" & & & & train & test & gap & train & test & gap & & \\\\\n\\midrule")
for ds in ORDER:
    b_, k_ = base.loc[ds], keep.loc[ds]
    row = v[v.dataset == ds].iloc[0]
    L.append(f"{tex(ds)} & {KIND[ds]} & {f4(b_.tree_ceiling)} & {f4(raw.loc[ds,'test'])} & "
             f"{f4(b_.train)} & {f4(b_.test)} & {f4(b_.gap)} & "
             f"{f4(k_.train)} & {f4(k_.test)} & {f4(k_.gap)} & {sgn(k_.test-b_.test)} & {row.holds}\\\\")
L.append("\\bottomrule\n\\end{tabular}}")
L.append("")

# --- per-dataset tables ---
for ds in ORDER:
    g = t[t.dataset == ds]
    is_clf = g.task_type.iloc[0] == "classification"
    L.append(f"% ---- {ds} (generated) ----")
    L.append(f"\\newcommand{{\\tab{ds.replace('_','').replace('16H','SixteenH')}}}{{%")
    if is_clf:
        L.append("\\begin{tabular}{lrrrrrrrrr}\n\\toprule")
        L.append("\\textbf{Arm} & \\textbf{Bits} & \\textbf{Flip} & "
                 "\\multicolumn{3}{c}{\\textbf{Ranking score}} & \\textbf{Gap} & "
                 "\\multicolumn{2}{c}{\\textbf{Accuracy}} & \\textbf{Log loss}\\\\")
        L.append("\\cmidrule(lr){4-6}\\cmidrule(lr){8-9}")
        L.append(" & & rate & train & val & test & & train & test & test\\\\\n\\midrule")
    else:
        L.append("\\begin{tabular}{lrrrrrrrrr}\n\\toprule")
        L.append("\\textbf{Arm} & \\textbf{Bits} & \\textbf{Flip} & "
                 "\\multicolumn{3}{c}{\\textbf{Explained variance}} & \\textbf{Gap} & "
                 "\\multicolumn{2}{c}{\\textbf{Typical error}} & \\textbf{Avg.\\ abs.\\ error}\\\\")
        L.append("\\cmidrule(lr){4-6}\\cmidrule(lr){8-9}")
        L.append(" & & rate & train & val & test & & train & test & test\\\\\n\\midrule")
    rx = g[g.view == "x"].iloc[0]
    if is_clf:
        L.append(f"raw features only (no tree bits) & {int(rx.bits)} & -- & {f4(rx.train)} & {f4(rx.val)} & {f4(rx.test)} & "
                 f"{sgn(rx.gap)} & {f4(rx.train_accuracy)} & {f4(rx.test_accuracy)} & {f4(rx.test_logloss)}\\\\")
    else:
        L.append(f"raw features only (no tree bits) & {int(rx.bits)} & -- & {f4(rx.train)} & {f4(rx.val)} & {f4(rx.test)} & "
                 f"{sgn(rx.gap)} & {rx.train_rmse:.3f} & {rx.test_rmse:.3f} & {rx.test_mae:.3f}\\\\")
    L.append("\\midrule")
    for arm in ARMS:
        r = g[(g.view == "tree") & (g.arm == arm)].iloc[0]
        bold = "\\textbf" if arm == "40_keep_L012" else ""
        name = f"{bold}{{{NAME[arm]}}}" if bold else NAME[arm]
        fl = "--" if r.mean_p == 0 else f"{r.mean_p:.3f}"
        if is_clf:
            L.append(f"{name} & {int(r.bits)} & {fl} & {f4(r.train)} & {f4(r.val)} & {f4(r.test)} & "
                     f"{sgn(r.gap)} & {f4(r.train_accuracy)} & {f4(r.test_accuracy)} & {f4(r.test_logloss)}\\\\")
        else:
            L.append(f"{name} & {int(r.bits)} & {fl} & {f4(r.train)} & {f4(r.val)} & {f4(r.test)} & "
                     f"{sgn(r.gap)} & {r.train_rmse:.3f} & {r.test_rmse:.3f} & {r.test_mae:.3f}\\\\")
    L.append("\\midrule")
    b_ = base.loc[ds]
    L.append(f"best tree model ({b_.best_tree.replace('_',' ')}) & & & & & {f4(b_.tree_ceiling)} & & & & \\\\")
    L.append("\\bottomrule\n\\end{tabular}}")
    L.append("")

# --- dataset description table pieces (sizes) ---
L.append("% ---- sizes (generated) ----")
L.append("\\newcommand{\\tabsizes}{%")
L.append("\\begin{tabular}{llrrrrrr}\n\\toprule")
L.append("\\textbf{Dataset} & \\textbf{Kind} & \\textbf{Raw features} & \\textbf{Train} & \\textbf{Val} & \\textbf{Test} & \\textbf{Tree bits} & \\textbf{Best tree}\\\\\n\\midrule")
import json, glob
for ds in ORDER:
    s = json.load(open(glob.glob(os.path.join(ROOT, ds, "00_baseline", "featreg_*.json"))[0]))
    L.append(f"{tex(ds)} & {KIND[ds]} & {s['n_raw_features']} & {s['n_train']:,} & {s['n_val']:,} & {s['n_test']:,} & "
             f"{s['n_bits_total']:,} & {s['best_tree'].replace('_',' ')} {f4(s['tree_ceiling'])}\\\\")
L.append("\\bottomrule\n\\end{tabular}}")
open(os.path.join(OUT, "tables.tex"), "w").write("\n".join(L))

# --- numbers used in the prose, as macros, so prose cannot drift from data ---
M = []
def mac(name, val): M.append(f"\\newcommand{{\\{name}}}{{{val}}}")
holds = int((v.holds == "yes").sum())
mac("nHolds", holds); mac("nDatasets", len(ORDER))
for ds in ORDER:
    key = ds.replace("_", "").replace("16H", "SixteenH")
    b_, k_ = base.loc[ds], keep.loc[ds]
    mac(f"{key}BaseTest", f4(b_.test)); mac(f"{key}BaseTrain", f4(b_.train)); mac(f"{key}BaseGap", f4(b_.gap))
    mac(f"{key}KeepTest", f4(k_.test)); mac(f"{key}KeepTrain", f4(k_.train)); mac(f"{key}KeepGap", f4(k_.gap))
    mac(f"{key}Tree", f4(b_.tree_ceiling)); mac(f"{key}Raw", f4(raw.loc[ds, "test"]))
    mac(f"{key}KeepBits", int(k_.bits)); mac(f"{key}Bits", int(b_.bits))
    mac(f"{key}DTest", sgn(k_.test - b_.test))
    mac(f"{key}GapCut", f"{(1 - k_.gap / b_.gap) * 100:.0f}" if b_.gap > 0.005 else "n/a")
    span = b_.tree_ceiling - raw.loc[ds, "test"]
    mac(f"{key}Closed", f"{(b_.test - raw.loc[ds,'test']) / span * 100:.0f}")
    for arm, tag in [("20_ramp_0.2", "Ramp"), ("21_ctrl_0.16", "Ctrl"), ("10_unif_0.2", "Unif"),
                     ("30_drop_L5", "DropFive"), ("31_drop_L45", "DropFourFive"), ("33_drop_L45_ramp02", "Combo")]:
        r = tree[(tree.dataset == ds) & (tree.arm == arm)].iloc[0]
        mac(f"{key}{tag}Test", f4(r.test)); mac(f"{key}{tag}Gap", f4(r.gap))
    if b_.task_type == "classification":
        mac(f"{key}BaseTrainAcc", f4(b_.train_accuracy)); mac(f"{key}KeepTrainAcc", f4(k_.train_accuracy))
        mac(f"{key}BaseTestAcc", f4(b_.test_accuracy)); mac(f"{key}KeepTestAcc", f4(k_.test_accuracy))
        mac(f"{key}BaseLL", f4(b_.test_logloss)); mac(f"{key}KeepLL", f4(k_.test_logloss))
        mac(f"{key}UnifLL", f4(tree[(tree.dataset == ds) & (tree.arm == "10_unif_0.2")].test_logloss.iloc[0]))
open(os.path.join(OUT, "numbers.tex"), "w").write("\n".join(M))
print(f"tables.tex ({len(L)} lines), numbers.tex ({len(M)} macros), 5 figures -> {OUT}")
print("holds:", holds, "of", len(ORDER))
