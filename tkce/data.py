"""Dataset loading for the Grinsztajn (2022) tabular benchmark via OpenML.

We drive everything off the four OpenML benchmark suites from
Grinsztajn, Oyallon & Varoquaux, "Why do tree-based models still outperform
deep learning on typical tabular data?" (NeurIPS 2022, arXiv 2207.08815):

    337  numerical   classification   (16 tasks)
    336  numerical   regression       (19 tasks)
    334  categorical classification   ( 7 tasks)
    335  categorical regression       (17 tasks)

A `Dataset` bundles a fully preprocessed, chronology-free train/val/test split
plus the metadata (task type, categorical mask, target stats) every downstream
model needs. Preprocessing decisions:

  * numerical features        -> StandardScaler, fit on TRAIN only
  * categorical features      -> ordinal codes (for trees / embeddings) AND a
                                 one-hot block (for the pure-MLP path)
  * classification target     -> integer labels 0..C-1
  * regression target         -> standardized (mean/std stored so predictions
                                 and RMSE can be reported on the original scale)

Splits are stratified for classification. Everything is cached to disk so the
first OpenML fetch is the only slow one.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

SUITES = {
    337: ("classification", "numerical"),
    336: ("regression", "numerical"),
    334: ("classification", "categorical"),
    335: ("regression", "categorical"),
}

CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "_cache")
os.makedirs(CACHE_DIR, exist_ok=True)


# --------------------------------------------------------------------------- #
# Suite registry
# --------------------------------------------------------------------------- #
def list_suite_tasks(refresh: bool = False) -> pd.DataFrame:
    """Return a dataframe of every (task_id, dataset, task_type, feature_type).

    Cached to results/_cache/suite_registry.csv after the first OpenML call.
    """
    path = os.path.join(CACHE_DIR, "suite_registry.csv")
    if os.path.exists(path) and not refresh:
        return pd.read_csv(path)

    import openml

    rows = []
    for sid, (task_type, feat_type) in SUITES.items():
        suite = openml.study.get_suite(sid)
        for tid in suite.tasks:
            try:
                t = openml.tasks.get_task(
                    tid, download_data=False, download_qualities=False,
                    download_splits=False, download_features_meta_data=False,
                )
                name = t.get_dataset(download_data=False).name
            except Exception as e:  # noqa: BLE001
                name = f"<err:{type(e).__name__}>"
            rows.append({"suite": sid, "task_id": tid, "dataset": name,
                         "task_type": task_type, "feature_type": feat_type})
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    return df


# --------------------------------------------------------------------------- #
# Dataset container
# --------------------------------------------------------------------------- #
@dataclass
class Dataset:
    name: str
    task_type: str                    # "classification" | "regression"
    # Numerical-encoded, scaled feature matrices (for trees & embedding path).
    X_train: np.ndarray
    X_val: np.ndarray
    X_test: np.ndarray
    # One-hot expanded feature matrices (for the plain-MLP-on-raw path).
    Xoh_train: np.ndarray
    Xoh_val: np.ndarray
    Xoh_test: np.ndarray
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray
    cat_mask: np.ndarray              # bool per column of X_* (ordinal-encoded)
    n_classes: int                    # 1 for regression
    cat_cardinalities: list = field(default_factory=list)  # #levels per cat col
    y_mean: float = 0.0               # regression target destandardization
    y_std: float = 1.0
    meta: dict = field(default_factory=dict)

    @property
    def n_features(self) -> int:
        return self.X_train.shape[1]

    @property
    def n_features_oh(self) -> int:
        return self.Xoh_train.shape[1]


# --------------------------------------------------------------------------- #
# Loading + preprocessing
# --------------------------------------------------------------------------- #
def load_task(task_id: int, seed: int = 0, val_frac: float = 0.15,
              test_frac: float = 0.15, max_rows: int | None = None) -> Dataset:
    """Fetch an OpenML task and return a fully preprocessed `Dataset`."""
    import openml

    task = openml.tasks.get_task(task_id, download_splits=False)
    ds = task.get_dataset()
    target = ds.default_target_attribute
    X, y, cat_ind, names = ds.get_data(target=target)

    is_clf = task.task_type_id == openml.tasks.TaskType.SUPERVISED_CLASSIFICATION
    task_type = "classification" if is_clf else "regression"

    df = X.copy()
    cat_mask = np.asarray(cat_ind, dtype=bool)

    # Drop rows with missing target; simple-impute feature NaNs.
    keep = y.notna().to_numpy()
    df, y = df.loc[keep].reset_index(drop=True), y.loc[keep].reset_index(drop=True)

    num_cols = [c for c, is_cat in zip(names, cat_mask) if not is_cat]
    cat_cols = [c for c, is_cat in zip(names, cat_mask) if is_cat]

    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
        df[c] = df[c].fillna(df[c].median())
    for c in cat_cols:
        df[c] = df[c].astype("object").fillna("__nan__").astype(str)

    if max_rows is not None and len(df) > max_rows:
        idx = np.random.default_rng(seed).choice(len(df), max_rows, replace=False)
        df, y = df.iloc[idx].reset_index(drop=True), y.iloc[idx].reset_index(drop=True)

    # Target encoding.
    if is_clf:
        classes, y_enc = np.unique(y.to_numpy(), return_inverse=True)
        n_classes = len(classes)
        y_mean, y_std = 0.0, 1.0
    else:
        y_enc = pd.to_numeric(y, errors="coerce").to_numpy(dtype=np.float64)
        n_classes = 1
        y_mean, y_std = float(np.mean(y_enc)), float(np.std(y_enc) + 1e-12)

    # --- Split (stratified for classification) ---
    idx_all = np.arange(len(df))
    strat = y_enc if is_clf else None
    idx_tr, idx_tmp = train_test_split(
        idx_all, test_size=val_frac + test_frac, random_state=seed, stratify=strat)
    strat_tmp = y_enc[idx_tmp] if is_clf else None
    rel = test_frac / (val_frac + test_frac)
    idx_va, idx_te = train_test_split(
        idx_tmp, test_size=rel, random_state=seed, stratify=strat_tmp)

    # --- Feature encoding, fit on TRAIN only ---
    # (a) numerical-encoded matrix: numerics scaled, categoricals -> ordinal ints.
    ord_enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    scaler = StandardScaler()

    def build_num(sub_idx, fit=False):
        parts = []
        if num_cols:
            xn = df.loc[sub_idx, num_cols].to_numpy(dtype=np.float64)
            xn = scaler.fit_transform(xn) if fit else scaler.transform(xn)
            parts.append(xn)
        if cat_cols:
            xc = df.loc[sub_idx, cat_cols].to_numpy()
            xc = ord_enc.fit_transform(xc) if fit else ord_enc.transform(xc)
            parts.append(xc.astype(np.float64))
        return np.concatenate(parts, axis=1).astype(np.float32) if parts \
            else np.zeros((len(sub_idx), 0), np.float32)

    X_train = build_num(idx_tr, fit=True)
    X_val = build_num(idx_va)
    X_test = build_num(idx_te)
    # Column order after concat: [num_cols..., cat_cols...]
    col_cat_mask = np.array([False] * len(num_cols) + [True] * len(cat_cols), dtype=bool)
    cat_cardinalities = [len(c) for c in ord_enc.categories_] if cat_cols else []

    # (b) one-hot matrix for the plain MLP path.
    oh = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

    def build_oh(sub_idx, fit=False):
        parts = []
        if num_cols:
            xn = df.loc[sub_idx, num_cols].to_numpy(dtype=np.float64)
            xn = scaler.transform(xn)  # scaler already fit above
            parts.append(xn)
        if cat_cols:
            xc = df.loc[sub_idx, cat_cols].to_numpy()
            xc = oh.fit_transform(xc) if fit else oh.transform(xc)
            parts.append(xc)
        return np.concatenate(parts, axis=1).astype(np.float32) if parts \
            else np.zeros((len(sub_idx), 0), np.float32)

    Xoh_train = build_oh(idx_tr, fit=True)
    Xoh_val = build_oh(idx_va)
    Xoh_test = build_oh(idx_te)

    def enc_y(sub_idx):
        yy = y_enc[sub_idx]
        if not is_clf:
            yy = (yy - y_mean) / y_std
        return yy.astype(np.int64 if is_clf else np.float32)

    return Dataset(
        name=ds.name, task_type=task_type,
        X_train=X_train, X_val=X_val, X_test=X_test,
        Xoh_train=Xoh_train, Xoh_val=Xoh_val, Xoh_test=Xoh_test,
        y_train=enc_y(idx_tr), y_val=enc_y(idx_va), y_test=enc_y(idx_te),
        cat_mask=col_cat_mask, n_classes=n_classes,
        cat_cardinalities=cat_cardinalities, y_mean=y_mean, y_std=y_std,
        meta={"task_id": task_id, "n_num": len(num_cols), "n_cat": len(cat_cols),
              "n_rows": len(df), "seed": seed},
    )
