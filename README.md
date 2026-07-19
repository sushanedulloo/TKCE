# TKCE — Tree-Kernel Contrastive Embeddings for Tabular Data

Can we make neural networks (MLP / TabResNet) as good as gradient-boosted trees
(XGBoost / CatBoost / LightGBM / Random Forest) on tabular data by **transferring
the tree inductive bias into the network's representation**?

A tree ensemble defines *what is similar* (two rows are similar if they land in
the same leaf across many trees — the leaf-co-occupancy kernel, which
approximates the Laplace kernel; Balog et al. 2016). A **Siamese encoder trained
with a contrastive objective** turns that similarity into a smooth embedding
space, and a downstream **MLP / TabResNet** does the actual prediction on top.

```
tabular X ──► tree kernel K(i,j) = (1/T) Σ_t 1[leaf_t(i)=leaf_t(j)]
                     │  (GBT / Random-Forest / unsupervised Mondrian)
                     ▼
        Siamese encoder φ, contrastive loss so ⟨φ(xᵢ),φ(xⱼ)⟩ ≈ K
                     │
        ┌────────────┴─────────────┐
   two-stage:                 joint (end-to-end):
   freeze φ, train head       train φ + head together with
   on φ(x) → y                task_loss + λ·contrastive_loss
```

We benchmark this against tree and NN baselines on the **Grinsztajn et al. (2022)**
tabular suite ("Why do tree-based models still outperform deep learning on
typical tabular data?", arXiv:2207.08815).

---

## Repository layout

| Path | What it is |
|---|---|
| `tkce/data.py` | OpenML loader for the Grinsztajn benchmark suites; preprocessing, splits, caching |
| `tkce/kernels.py` | `GBTKernel`, `RFKernel`, unsupervised `MondrianKernel` (leaf-co-occupancy) |
| `tkce/models.py` | `SiameseEncoder`, `MLPHead`, `TabResNet` (Gorishniy 2021), `JointModel` |
| `tkce/losses.py` | `InfoNCELoss`, `KernelRegressionLoss` |
| `tkce/pairs.py` | Scalable sampled positive-pair mining from the kernel |
| `tkce/train.py` | `pretrain_encoder` (stage A), `train_head` (stage B / baselines), `train_joint` (end-to-end) |
| `tkce/baselines.py` | XGBoost / LightGBM / CatBoost / RF + leaf-one-hot→MLP (He et al. 2014) |
| `tkce/metrics.py` | Accuracy + ROC-AUC (clf); RMSE + R² (reg) |
| `tkce/tuning.py` | Optuna search spaces + the 12-model registry |
| `tkce/aggregate.py` | Per-dataset ranks, average rank, Wilcoxon, critical-difference diagram |
| `run_slice.py` | **One dataset → all model cells → table** (fast sanity harness) |
| `run_suite.py` | **Full protocol: tune → multi-seed → aggregate** across datasets |
| `knowledge/` | Theory notes (tree kernel, contrastive learning, references) |
| `paper/` | LaTeX: the AAAI draft and a plain-language experiments overview |
| `colab_run.ipynb` | Resumable Colab Pro+ notebook to run the whole thing |

---

## Installation

```bash
git clone <your-repo-url> tree-kernel-contrastive
cd tree-kernel-contrastive

conda create -n tkce python=3.11 -y
conda activate tkce
pip install -r requirements-tkce.txt
```

> **PyTorch:** install the build that matches your machine (CUDA on the GPU
> server) from https://pytorch.org/get-started/locally/. The first OpenML fetch
> downloads and caches the datasets.

---

## Quick start — sanity check on one dataset

Runs every model cell on a single dataset and prints a ranked table (a few
minutes, no tuning):

```bash
python run_slice.py --task 361065 --max-rows 4000 --device cpu
```

Output → console table + `results/slice_<dataset>_seed0.csv`. Good for verifying
the install and eyeballing whether the effect exists on a given dataset.

`run_slice.py` arguments: `--task <openml_task_id>` · `--max-rows N` (cap rows
for speed) · `--seed S` · `--device auto|cpu|cuda|mps`.

---

## Run on Colab Pro+ (recommended for the full run)

Open **`colab_run.ipynb`** in Colab (GPU T4/L4 + High-RAM, background execution on),
set your GitHub URL in the config cell, and run top to bottom. It mounts Google
Drive, clones the repo, installs deps, runs the full 15-dataset suite, builds the
report, and runs the analysis experiments — **saving everything to Drive as it
goes**. The suite is **resumable**: if the session dies, re-run the clone + full-run
cells and it continues from where it stopped (finished dataset×model cells are
skipped). See `paper/experiments_overview.tex` for a plain-language description of
every experiment.

## Full experiments

The paper protocol: for each dataset, tune every model's hyperparameters with an
**equal Optuna budget** on the seed-0 split, then evaluate the winning config on
every seed's test split, then aggregate (ranks / CD diagram / Wilcoxon).

```bash
python run_suite.py \
  --tasks 361070,361062,361063,361072,361076,361077 \
  --seeds 0,1,2 \
  --trials 100 \
  --max-rows 6000 \
  --device auto \
  --reference catboost \
  --out results/suite_v1
```

### `run_suite.py` arguments

| Flag | Default | Meaning |
|---|---|---|
| `--tasks` | 3 clf datasets | Comma-separated OpenML **task** ids |
| `--seeds` | `0,1,2` | Random seeds (splits); more = tighter error bars |
| `--trials` | `100` | Optuna trials **per model** (equal budget) |
| `--max-rows` | `8000` | Row cap per dataset (raise/remove on a big GPU) |
| `--device` | `mps` | `auto` picks cuda→mps→cpu |
| `--models` | `all` | Subset, e.g. `xgboost,mlp_raw,tkce_joint_gbt_mlp` |
| `--reference` | `catboost` | Baseline for the Wilcoxon comparison |
| `--out` | `results/suite_run` | Output directory |

### The 16 models

| Group | Models |
|---|---|
| Tree baselines | `xgboost`, `lightgbm`, `catboost`, `random_forest` |
| NN baselines (raw features) | `mlp_raw`, `tabresnet_raw` |
| Strong deep baselines | `ft_transformer`, `num_embed_mlp` (periodic embeddings, arXiv 2203.05556) |
| Tree-feature baselines | `leafonehot_mlp` (He et al. 2014); `pca_gbt_mlp`, `pca_gbt_tabresnet` (PCA of leaf features — linear counterpart to TKCE) |
| **TKCE two-stage** | `tkce2s_gbt_mlp`, `tkce2s_gbt_tabresnet`, `tkce2s_mondrian_mlp` |
| **TKCE joint** | `tkce_joint_gbt_mlp`, `tkce_joint_gbt_tabresnet` |

### Outputs (in `--out`)

| File | Contents |
|---|---|
| `results_long.csv` | One row per dataset × model × seed (the raw results) |
| `pivot_mean.csv` | Mean primary metric per dataset × model |
| `ranks.csv`, `avg_rank.csv` | Per-dataset ranks and average rank per model |
| `wilcoxon.csv` | Signed-rank test of each model vs `--reference` |
| `cd_diagram.png` | Critical-difference (Nemenyi) diagram |
| `config.json`, `run.log` | Run config and log |

### Monitoring & resuming

`results_long.csv` is written after **every** model, so it is the progress
signal (target rows = `models × seeds × datasets`):

```bash
wc -l results/suite_v1/results_long.csv
```

The run is **resumable** — re-running the same command skips any dataset×model
cell already complete for all requested seeds. Safe to interrupt.

### Generate the paper report

Turn a finished run's `results_long.csv` into paper-ready tables + figures:

```bash
python make_report.py --results results/suite_v1/results_long.csv \
    --out results/suite_v1/report --reference catboost
```

Produces (in `--out`): `report.md`, the main results table, **gap analysis**
(% of the tree→NN gap closed), **two-stage vs joint**, **kernel ablation**
(GBT vs Mondrian), average-rank table + `cd_diagram.png`, and Wilcoxon tests.

### Paper draft

`paper/methodology_experiments.tex` is the AAAI Methodology + Experiments draft
(compiles standalone; also drops into the AAAI template). Conceptual diagrams are
marked placeholders with copy-paste **image-generation prompts**; result figures
come straight from code (`make_report.py` and the analysis scripts below).

### Analysis experiments (the *why*/*when* figures)

Three controlled experiments back the analysis section. Each writes a figure to
`paper/figures/` and a CSV to `results/analysis/`:

```bash
# (1) Mechanism probes — noise robustness, rotation sensitivity, irregular targets
python run_mechanism.py --task 361065 --seeds 0,1,2 --device auto

# (2) Data-efficiency curves — metric vs. training-set fraction
python run_data_efficiency.py --tasks 361070,361072 --seeds 0,1,2 --device auto

# (3) Lambda sweep — the joint regime's contrastive weight
python run_lambda_sweep.py --tasks 361070 --seeds 0,1,2 --device auto

# (4) Contrastive-loss ablation — which loss learns the best embedding
python run_loss_ablation.py --tasks 361070,361072 --seeds 0,1,2 --device auto
```

The **7 contrastive losses** (`tkce.losses.ALL_LOSSES`) are: `infonce`,
`kernel_regression`, `contrastive` (Hadsell), `triplet` (FaceNet), `supcon`
(multi-positive), `aninfonce` (anisotropic), `clip_infonce` (symmetric +
learnable temperature). Select one anywhere via `contrastive_loss` in the config;
two-stage supports all seven, the joint regime supports the in-batch family.

These sweep a compact 4-model panel (`tkce.analysis.MODELS_DEFAULT`: XGBoost,
raw MLP, MLP+num-embed, TKCE-joint) with a light training budget and produce
`mechanism.png`, `data_efficiency.png`, `lambda_sweep.png` (referenced directly
by the LaTeX).

---

## Benchmark datasets (Grinsztajn 2022, via OpenML)

Everything is driven off four OpenML benchmark suites — list all tasks with:

```python
from tkce.data import list_suite_tasks
print(list_suite_tasks())      # cached to results/_cache/suite_registry.csv
```

| Suite | Content | #tasks |
|---|---|---|
| 337 | numerical classification | 16 |
| 336 | numerical regression | 19 |
| 334 | categorical classification | 7 |
| 335 | categorical regression | 17 |

Default subset used above: `eye_movements (361070)`, `pol (361062)`,
`house_16H (361063)` (clf) and `cpu_act (361072)`, `wine_quality (361076)`,
`Ailerons (361077)` (reg) — chosen for a real tree > NN gap.

---

## Reproducing specific analyses

- **Two-stage vs joint** — compare `tkce2s_gbt_*` against `tkce_joint_gbt_*` in
  `pivot_mean.csv` / `avg_rank.csv`.
- **Kernel ablation (supervised vs unsupervised)** — compare `tkce2s_gbt_mlp`
  (target-led GBT kernel) against `tkce2s_mondrian_mlp` (label-free Mondrian).
- **Does contrastive beat raw leaf features?** — TKCE cells vs `leafonehot_mlp`.
- **Is the gap real?** — best tree vs `mlp_raw` / `tabresnet_raw` per dataset.

---

## Extending

- **New dataset:** pass its OpenML task id to `--tasks`. (Categorical suites
  334/335 load via the same path; validate before large runs.)
- **New model:** add a `suggest`/`run` pair to `MODEL_SPECS` in `tkce/tuning.py`.
- **New kernel:** subclass `_LeafKernel` in `tkce/kernels.py` and register it in
  `KERNELS`.

---

## Performance notes

- **Device:** the nets are small, so on a laptop **CPU can beat MPS**; on a CUDA
  server the GPU helps more, but the tree baselines run on CPU regardless. Quick
  check with `--device cuda` vs `--device cpu` on one dataset.
- **Scaling:** pair mining is sampled (O(pairs·T)), and pretraining caps mining
  rows (`max_pretrain_rows`), so datasets of tens of thousands of rows are fine.

---

## References

- Grinsztajn, Oyallon, Varoquaux (2022). *Why do tree-based models still
  outperform deep learning on typical tabular data?* NeurIPS D&B. arXiv:2207.08815.
- Gorishniy, Rubachev, Babenko (2022). *On Embeddings for Numerical Features in
  Tabular Deep Learning.* arXiv:2203.05556. (numerical-embedding baseline)
- Gorishniy et al. (2021). *Revisiting Deep Learning Models for Tabular Data.*
  arXiv:2106.11959. (TabResNet)
- Balog et al. (2016). *The Mondrian Kernel.* UAI. arXiv:1606.05241.
- He et al. (2014). *Practical Lessons from Predicting Clicks on Ads at Facebook.*
  (GBDT leaf-one-hot features)
- van den Oord et al. (2018). *Representation Learning with Contrastive Predictive
  Coding* (InfoNCE). arXiv:1807.03748.
- See `knowledge/04_references.md` for the full list and code mapping.
