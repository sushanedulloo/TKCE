"""Downstream-task metrics, computed on the ORIGINAL target scale.

Classification: accuracy + ROC-AUC (binary or one-vs-rest macro for multiclass).
Regression:     RMSE + R^2, after de-standardizing predictions/targets.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, r2_score, roc_auc_score


def clf_metrics(y_true: np.ndarray, proba: np.ndarray) -> dict:
    """proba: (n, C) class probabilities."""
    pred = proba.argmax(axis=1)
    acc = accuracy_score(y_true, pred)
    try:
        if proba.shape[1] == 2:
            auc = roc_auc_score(y_true, proba[:, 1])
        else:
            auc = roc_auc_score(y_true, proba, multi_class="ovr", average="macro")
    except Exception:  # noqa: BLE001 - degenerate (single class in split)
        auc = float("nan")
    return {"accuracy": float(acc), "auc": float(auc)}


def reg_metrics(y_true: np.ndarray, pred: np.ndarray,
                y_mean: float = 0.0, y_std: float = 1.0) -> dict:
    """y_true/pred are standardized; report on the original scale."""
    yt = y_true * y_std + y_mean
    yp = pred * y_std + y_mean
    rmse = float(np.sqrt(np.mean((yt - yp) ** 2)))
    return {"rmse": rmse, "r2": float(r2_score(yt, yp))}


def score(dataset, y_true, out) -> dict:
    """Dispatch on task type. `out` is proba (clf) or standardized pred (reg)."""
    if dataset.task_type == "classification":
        return clf_metrics(y_true, out)
    return reg_metrics(y_true, out, dataset.y_mean, dataset.y_std)


def primary_metric(task_type: str) -> str:
    return "auc" if task_type == "classification" else "r2"
