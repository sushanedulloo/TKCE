"""Tree-model baselines and the GBDT-leaf-one-hot feature baseline.

Tree models (the bar our NNs must reach): XGBoost, CatBoost, LightGBM, RF.
They consume the numerical-encoded matrix (categoricals as ordinal ints).

leaf_onehot_features: the classic He et al. 2014 (Facebook) trick — feed the
GBDT's leaf indices, one-hot encoded, as features to a linear/MLP model. This
is the CHEAP alternative to our contrastive embedding, so our method must beat
it to justify the machinery.
"""

from __future__ import annotations

import numpy as np

from .metrics import score


def _predict(model, X, task_type):
    if task_type == "classification":
        return model.predict_proba(X)
    return model.predict(X)


def fit_tree_baseline(name, dataset, cfg=None):
    """Fit one tree model on train, return (test_metrics, val_metrics, model)."""
    cfg = cfg or {}
    tt = dataset.task_type
    Xtr, ytr = dataset.X_train, dataset.y_train
    is_clf = tt == "classification"

    if name == "xgboost":
        import xgboost as xgb
        params = dict(n_estimators=cfg.get("n_estimators", 300),
                      max_depth=cfg.get("max_depth", 6),
                      learning_rate=cfg.get("learning_rate", 0.1),
                      subsample=0.8, colsample_bytree=0.8,
                      n_jobs=-1, tree_method="hist", random_state=cfg.get("seed", 0))
        model = (xgb.XGBClassifier(**params, eval_metric="logloss")
                 if is_clf else xgb.XGBRegressor(**params))
    elif name == "lightgbm":
        import lightgbm as lgb
        params = dict(n_estimators=cfg.get("n_estimators", 300),
                      max_depth=cfg.get("max_depth", -1),
                      learning_rate=cfg.get("learning_rate", 0.05),
                      subsample=0.8, colsample_bytree=0.8,
                      n_jobs=-1, random_state=cfg.get("seed", 0), verbose=-1)
        model = (lgb.LGBMClassifier(**params) if is_clf else lgb.LGBMRegressor(**params))
    elif name == "catboost":
        from catboost import CatBoostClassifier, CatBoostRegressor
        params = dict(iterations=cfg.get("n_estimators", 500),
                      depth=cfg.get("max_depth", 6),
                      learning_rate=cfg.get("learning_rate", 0.05),
                      random_seed=cfg.get("seed", 0), verbose=False,
                      allow_writing_files=False)
        model = (CatBoostClassifier(**params) if is_clf else CatBoostRegressor(**params))
    elif name == "random_forest":
        from sklearn.ensemble import (RandomForestClassifier,
                                      RandomForestRegressor)
        params = dict(n_estimators=cfg.get("n_estimators", 300),
                      max_depth=cfg.get("max_depth", None),
                      n_jobs=-1, random_state=cfg.get("seed", 0))
        model = (RandomForestClassifier(**params) if is_clf
                 else RandomForestRegressor(**params))
    else:
        raise ValueError(f"unknown tree baseline '{name}'")

    model.fit(Xtr, ytr)
    val = score(dataset, dataset.y_val, _predict(model, dataset.X_val, tt))
    test = score(dataset, dataset.y_test, _predict(model, dataset.X_test, tt))
    return test, val, model


def leaf_onehot_features(dataset, cfg=None):
    """Return one-hot leaf-index features for train/val/test from an XGBoost fit."""
    from sklearn.preprocessing import OneHotEncoder
    import xgboost as xgb
    cfg = cfg or {}
    is_clf = dataset.task_type == "classification"
    params = dict(n_estimators=cfg.get("n_estimators", 200),
                  max_depth=cfg.get("max_depth", 4), learning_rate=0.1,
                  subsample=0.8, colsample_bytree=0.8, n_jobs=-1,
                  tree_method="hist", random_state=cfg.get("seed", 0))
    m = (xgb.XGBClassifier(**params, eval_metric="logloss")
         if is_clf else xgb.XGBRegressor(**params))
    m.fit(dataset.X_train, dataset.y_train)
    oh = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    Ltr = oh.fit_transform(m.apply(dataset.X_train)).astype(np.float32)
    Lva = oh.transform(m.apply(dataset.X_val)).astype(np.float32)
    Lte = oh.transform(m.apply(dataset.X_test)).astype(np.float32)
    return Ltr, Lva, Lte


def leaf_pca_features(dataset, cfg=None):
    """PCA-reduced leaf-one-hot features.

    The leaf one-hot vector is the tree kernel's explicit feature map
    (<z_i, z_j> = K), so PCA on it is the LINEAR spectral embedding of the tree
    kernel — the non-contrastive counterpart to the TKCE embedding. PCA is fit on
    train and applied to val/test, so the embedding is inductive (unlike a
    transductive kernel-PCA on the Gram matrix)."""
    from sklearn.decomposition import PCA
    cfg = cfg or {}
    Ltr, Lva, Lte = leaf_onehot_features(dataset, cfg)
    n_comp = int(min(cfg.get("n_components", 64), Ltr.shape[1], Ltr.shape[0]))
    pca = PCA(n_components=n_comp, random_state=cfg.get("seed", 0)).fit(Ltr)
    return (pca.transform(Ltr).astype(np.float32),
            pca.transform(Lva).astype(np.float32),
            pca.transform(Lte).astype(np.float32))
