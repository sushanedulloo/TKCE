"""Helpers for the controlled analysis experiments.

Shared by run_mechanism.py, run_data_efficiency.py, run_lambda_sweep.py:

  * build synthetic datasets with a controllable target irregularity,
  * perturb any Dataset (add uninformative features, rotate the feature space,
    subsample the training set),
  * evaluate any registered model on a Dataset via tkce.tuning.run_model,
  * a small consistent plotting helper.

Perturbations return a NEW Dataset (via dataclasses.replace); the numerical
matrix X and the one-hot matrix Xoh are kept consistent so both the TKCE path
(uses X) and the raw-NN path (uses Xoh) see the same perturbation.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from .data import Dataset
from .metrics import primary_metric
from .tuning import run_model

# Compact model panel for the analysis figures: tree bar, raw NN, rival
# embedding, and our method.
MODELS_DEFAULT = ["xgboost", "mlp_raw", "num_embed_mlp", "tkce_joint_gbt_mlp"]

# Light training budget (analysis sweeps have many points; full tuning is
# unnecessary and slow). Callers can override.
ANALYSIS_CFG = dict(head_epochs=60, patience=10, pretrain_epochs=20)


# --------------------------------------------------------------------------- #
# Dataset construction / perturbation
# --------------------------------------------------------------------------- #
def make_dataset_from_arrays(X, y, task_type, seed=0, val_frac=0.15,
                             test_frac=0.15, name="synthetic") -> Dataset:
    X = np.asarray(X, dtype=np.float32)
    is_clf = task_type == "classification"
    if is_clf:
        _, y = np.unique(np.asarray(y), return_inverse=True)
        y = y.astype(np.int64)
        n_classes, y_mean, y_std = len(np.unique(y)), 0.0, 1.0
    else:
        y = np.asarray(y, dtype=np.float64)
        n_classes = 1
        y_mean, y_std = float(y.mean()), float(y.std() + 1e-12)
        y = ((y - y_mean) / y_std).astype(np.float32)

    idx = np.arange(len(X))
    strat = y if is_clf else None
    tr, tmp = train_test_split(idx, test_size=val_frac + test_frac,
                               random_state=seed, stratify=strat)
    strat_tmp = y[tmp] if is_clf else None
    va, te = train_test_split(tmp, test_size=test_frac / (val_frac + test_frac),
                              random_state=seed, stratify=strat_tmp)
    sc = StandardScaler().fit(X[tr])
    Xs = sc.transform(X).astype(np.float32)
    return Dataset(
        name=name, task_type=task_type,
        X_train=Xs[tr], X_val=Xs[va], X_test=Xs[te],
        Xoh_train=Xs[tr], Xoh_val=Xs[va], Xoh_test=Xs[te],
        y_train=y[tr], y_val=y[va], y_test=y[te],
        cat_mask=np.zeros(X.shape[1], bool), n_classes=n_classes,
        cat_cardinalities=[], y_mean=y_mean, y_std=y_std,
        meta={"n_num": X.shape[1], "n_cat": 0, "n_rows": len(X), "seed": seed})


def make_synthetic_irregular(n=4000, p=4, freq=2, seed=0):
    """Sine-based target: y = 1[sum_j sin(2*pi*freq*x_j) > median]. Higher freq =
    more oscillations = a more irregular decision boundary (trees fit it; smooth
    NNs struggle). Stays balanced and non-degenerate at every freq >= 1."""
    rng = np.random.default_rng(seed)
    X = rng.uniform(0.0, 1.0, size=(n, p))
    s = np.sin(2 * np.pi * freq * X).sum(axis=1)
    y = (s > np.median(s)).astype(int)
    return X.astype(np.float32), y


def add_noise_features(ds: Dataset, k: int, seed=0) -> Dataset:
    """Append k standard-normal (uninformative) feature columns."""
    if k <= 0:
        return ds
    rng = np.random.default_rng(seed + 999)

    def cat(a):
        return np.concatenate(
            [a, rng.standard_normal((a.shape[0], k)).astype(np.float32)], axis=1)
    Xtr, Xva, Xte = cat(ds.X_train), cat(ds.X_val), cat(ds.X_test)
    return replace(ds, X_train=Xtr, X_val=Xva, X_test=Xte,
                   Xoh_train=Xtr, Xoh_val=Xva, Xoh_test=Xte,
                   cat_mask=np.zeros(Xtr.shape[1], bool),
                   meta={**ds.meta, "n_num": ds.meta["n_num"] + k})


def rotate_features(ds: Dataset, angle_deg: float, seed=0) -> Dataset:
    """Rotate disjoint feature pairs by `angle_deg` (Givens rotations).

    angle 0 = axis-aligned (unchanged); 90 = fully rotated. The SAME rotation
    is applied to train/val/test. Only valid for all-numerical datasets."""
    if angle_deg == 0:
        return ds
    p = ds.X_train.shape[1]
    theta = np.radians(angle_deg)
    c, s = np.cos(theta), np.sin(theta)
    perm = np.random.default_rng(seed + 7).permutation(p)
    R = np.eye(p, dtype=np.float32)
    for a in range(0, p - 1, 2):
        i, j = perm[a], perm[a + 1]
        R[i, i] = c; R[i, j] = -s; R[j, i] = s; R[j, j] = c

    def rot(X):
        return (X @ R).astype(np.float32)
    Xtr, Xva, Xte = rot(ds.X_train), rot(ds.X_val), rot(ds.X_test)
    return replace(ds, X_train=Xtr, X_val=Xva, X_test=Xte,
                   Xoh_train=Xtr, Xoh_val=Xva, Xoh_test=Xte)


def subsample_train(ds: Dataset, frac: float, seed=0) -> Dataset:
    """Keep a stratified (clf) fraction of the training rows; val/test intact."""
    if frac >= 1.0:
        return ds
    n = len(ds.X_train)
    rng = np.random.default_rng(seed + 3)
    if ds.task_type == "classification":
        idx = []
        for cls in np.unique(ds.y_train):
            ci = np.where(ds.y_train == cls)[0]
            take = max(2, int(round(len(ci) * frac)))
            idx.extend(rng.choice(ci, min(take, len(ci)), replace=False))
        idx = np.array(idx)
    else:
        idx = rng.choice(n, max(20, int(n * frac)), replace=False)
    return replace(ds, X_train=ds.X_train[idx], Xoh_train=ds.Xoh_train[idx],
                   y_train=ds.y_train[idx],
                   meta={**ds.meta, "n_rows": len(idx) + len(ds.X_val) + len(ds.X_test)})


# --------------------------------------------------------------------------- #
# Evaluation + plotting
# --------------------------------------------------------------------------- #
def evaluate(model_name, ds, device, seed, cfg=None) -> float:
    """Return the primary test metric (AUC/R^2) for a model on a dataset."""
    c = {**ANALYSIS_CFG, **(cfg or {})}
    out = run_model(model_name, c, ds, device, seed)
    return float(out["test"][primary_metric(ds.task_type)])


PRETTY = {"xgboost": "XGBoost", "mlp_raw": "MLP (raw)",
          "num_embed_mlp": "MLP + num-embed", "tkce_joint_gbt_mlp": "TKCE (joint)",
          "tkce2s_gbt_mlp": "TKCE (two-stage)", "catboost": "CatBoost"}


def line_plot(ax, xs, series, xlabel, ylabel, title, logx=False):
    """series: {model_name: (mean_array, std_array)}."""
    markers = ["o", "s", "^", "D", "v", "P"]
    for i, (name, (mean, std)) in enumerate(series.items()):
        mean, std = np.asarray(mean), np.asarray(std)
        ax.plot(xs, mean, marker=markers[i % len(markers)], label=PRETTY.get(name, name))
        ax.fill_between(xs, mean - std, mean + std, alpha=0.15)
    if logx:
        ax.set_xscale("log")
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.set_title(title)
    ax.grid(True, alpha=0.3); ax.legend(fontsize=8)
