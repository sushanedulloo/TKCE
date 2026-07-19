"""Optuna hyperparameter tuning with an equal trial budget per model.

Every model is a spec with two functions:

    suggest(trial, dataset) -> cfg      # sample a hyperparameter config
    run(cfg, dataset, device, seed)     # -> {"val": <primary metric>, "test": {...}}

`tune_model` maximizes the validation primary metric (AUC for classification,
R^2 for regression) over `n_trials`, then returns the best cfg. `run_model`
re-runs a fixed cfg (used to evaluate the tuned config across seeds).

Protocol: tune on the seed-0 split, freeze the winning cfg, evaluate it on every
seed's test split. Trees are cheap; NN/TKCE cells dominate the cost, so their
search spaces keep the training budget bounded (early stopping + capped epochs).
"""

from __future__ import annotations

import warnings

import numpy as np

from .baselines import (fit_tree_baseline, leaf_onehot_features,
                        leaf_pca_features)
from .kernels import build_kernel
from .metrics import primary_metric, score
from .train import (encode, predict_head, predict_joint, pretrain_encoder,
                    train_head, train_joint)

warnings.filterwarnings("ignore")

BASE_HEAD = dict(head_epochs=120, patience=16, batch_size=256, weight_decay=1e-4)

# Fixed downstream-head config: held CONSTANT across mlp_raw / tabresnet_raw /
# leafonehot_mlp / all TKCE heads, so any difference comes from the INPUT
# representation, not from head tuning. Merged AFTER cfg so it always wins.
# (The strong deep baselines ft_transformer / num_embed_mlp are NOT heads on a
# representation, so they keep their own tuned architectures.)
FIXED_HEAD = dict(hidden_dims=(256, 128), dropout=0.1,          # MLP head
                  d=192, d_hidden=256, n_blocks=3,              # TabResNet head
                  head_lr=1e-3)


# --------------------------------------------------------------------------- #
# Per-model runners: return validation primary metric + full test metrics.
# --------------------------------------------------------------------------- #
def _pm(ds):
    return primary_metric(ds.task_type)


def _run_tree(name):
    def run(cfg, ds, device, seed):
        test, val, _ = fit_tree_baseline(name, ds, {**cfg, "seed": seed})
        return {"val": val[_pm(ds)], "test": test}
    return run


def _run_nn_raw(kind):
    def run(cfg, ds, device, seed):
        c = {**BASE_HEAD, **cfg, **FIXED_HEAD, "seed": seed}
        model = train_head(kind, ds.Xoh_train, ds.y_train, ds.Xoh_val, ds.y_val,
                           ds, c, device)
        val = score(ds, ds.y_val, predict_head(model, ds.Xoh_val, ds, device))
        test = score(ds, ds.y_test, predict_head(model, ds.Xoh_test, ds, device))
        return {"val": val[_pm(ds)], "test": test}
    return run


def _run_special(kind):
    """FT-Transformer / NumEmbedMLP: consume the ordinal-encoded X + cat info."""
    def run(cfg, ds, device, seed):
        c = {**BASE_HEAD, **cfg, "seed": seed, "n_num": ds.meta["n_num"],
             "cat_cardinalities": ds.cat_cardinalities}
        model = train_head(kind, ds.X_train, ds.y_train, ds.X_val, ds.y_val, ds, c, device)
        val = score(ds, ds.y_val, predict_head(model, ds.X_val, ds, device))
        test = score(ds, ds.y_test, predict_head(model, ds.X_test, ds, device))
        return {"val": val[_pm(ds)], "test": test}
    return run


def _run_leaf_onehot(cfg, ds, device, seed):
    c = {**BASE_HEAD, **cfg, **FIXED_HEAD, "seed": seed}
    Ltr, Lva, Lte = leaf_onehot_features(
        ds, {"n_estimators": cfg.get("k_n_estimators", 200),
             "max_depth": cfg.get("k_max_depth", 4), "seed": seed})
    model = train_head("mlp", Ltr, ds.y_train, Lva, ds.y_val, ds, c, device)
    val = score(ds, ds.y_val, predict_head(model, Lva, ds, device))
    test = score(ds, ds.y_test, predict_head(model, Lte, ds, device))
    return {"val": val[_pm(ds)], "test": test}


def _run_pca(head_kind):
    """PCA of the tree-leaf features -> FIXED head. The linear (non-contrastive)
    counterpart to TKCE: does contrastive learning beat plain PCA of the same
    tree features?"""
    def run(cfg, ds, device, seed):
        c = {**BASE_HEAD, **cfg, **FIXED_HEAD, "seed": seed}
        Ptr, Pva, Pte = leaf_pca_features(
            ds, {"n_estimators": cfg.get("k_n_estimators", 200),
                 "max_depth": cfg.get("k_max_depth", 4),
                 "n_components": cfg.get("n_components", 64), "seed": seed})
        model = train_head(head_kind, Ptr, ds.y_train, Pva, ds.y_val, ds, c, device)
        val = score(ds, ds.y_val, predict_head(model, Pva, ds, device))
        test = score(ds, ds.y_test, predict_head(model, Pte, ds, device))
        return {"val": val[_pm(ds)], "test": test}
    return run


def _run_tkce_twostage(kernel_name, head_kind):
    def run(cfg, ds, device, seed):
        c = {**BASE_HEAD, **cfg, **FIXED_HEAD, "seed": seed}
        kcfg = {"n_estimators": cfg.get("k_n_estimators", 200),
                "max_depth": cfg.get("k_max_depth", 4), "random_state": seed}
        kern = build_kernel(kernel_name, ds.X_train, ds.y_train, ds.task_type, kcfg)
        enc, _ = pretrain_encoder(ds.X_train, kern, c, device)
        etr, eva, ete = (encode(enc, ds.X_train, device),
                         encode(enc, ds.X_val, device),
                         encode(enc, ds.X_test, device))
        model = train_head(head_kind, etr, ds.y_train, eva, ds.y_val, ds, c, device)
        val = score(ds, ds.y_val, predict_head(model, eva, ds, device))
        test = score(ds, ds.y_test, predict_head(model, ete, ds, device))
        return {"val": val[_pm(ds)], "test": test}
    return run


def _run_tkce_joint(kernel_name, head_kind):
    def run(cfg, ds, device, seed):
        c = {**BASE_HEAD, **cfg, **FIXED_HEAD, "seed": seed}
        kcfg = {"n_estimators": cfg.get("k_n_estimators", 200),
                "max_depth": cfg.get("k_max_depth", 4), "random_state": seed}
        kern = build_kernel(kernel_name, ds.X_train, ds.y_train, ds.task_type, kcfg)
        model = train_joint(head_kind, ds.X_train, ds.y_train, ds.X_val, ds.y_val,
                            kern, ds, c, device)
        val = score(ds, ds.y_val, predict_joint(model, ds.X_val, ds, device))
        test = score(ds, ds.y_test, predict_joint(model, ds.X_test, ds, device))
        return {"val": val[_pm(ds)], "test": test}
    return run


# --------------------------------------------------------------------------- #
# Search spaces
# --------------------------------------------------------------------------- #
def _sg_xgboost(t, ds):
    return dict(n_estimators=t.suggest_int("n_estimators", 100, 1000, step=100),
                max_depth=t.suggest_int("max_depth", 2, 10),
                learning_rate=t.suggest_float("learning_rate", 1e-2, 0.3, log=True),
                subsample=t.suggest_float("subsample", 0.5, 1.0),
                colsample_bytree=t.suggest_float("colsample_bytree", 0.5, 1.0))


def _sg_lightgbm(t, ds):
    return dict(n_estimators=t.suggest_int("n_estimators", 100, 1000, step=100),
                max_depth=t.suggest_int("max_depth", -1, 12),
                learning_rate=t.suggest_float("learning_rate", 1e-2, 0.3, log=True))


def _sg_catboost(t, ds):
    return dict(n_estimators=t.suggest_int("n_estimators", 200, 1000, step=100),
                max_depth=t.suggest_int("max_depth", 3, 10),
                learning_rate=t.suggest_float("learning_rate", 1e-2, 0.3, log=True))


def _sg_rf(t, ds):
    return dict(n_estimators=t.suggest_int("n_estimators", 100, 800, step=100),
                max_depth=t.suggest_int("max_depth", 4, 30))


def _sg_nn(t, ds):
    """Search space for the strong deep baselines (ft/num_embed), which ARE the
    tuned architecture (not a fixed head on a representation)."""
    width = t.suggest_categorical("width", [128, 256, 384, 512])
    depth = t.suggest_int("depth", 1, 3)
    return dict(hidden_dims=tuple([width] * depth),
                d=width, d_hidden=width, n_blocks=t.suggest_int("n_blocks", 1, 4),
                dropout=t.suggest_float("dropout", 0.0, 0.4),
                head_lr=t.suggest_float("head_lr", 3e-4, 5e-3, log=True))


def _sg_fixed(t, ds):
    """Empty search space: model uses the FIXED_HEAD on raw features, nothing to
    tune (a controlled baseline). tune_model runs it once instead of N trials."""
    return {}


def _sg_ft(t, ds):
    return dict(d_token=t.suggest_categorical("d_token", [64, 128, 192]),
                n_blocks=t.suggest_int("n_blocks", 1, 4),
                n_heads=8,
                dropout=t.suggest_float("dropout", 0.0, 0.3),
                head_lr=t.suggest_float("head_lr", 1e-4, 3e-3, log=True),
                weight_decay=t.suggest_float("weight_decay", 1e-6, 1e-3, log=True))


def _sg_numembed(t, ds):
    c = _sg_nn(t, ds)
    c.update(k_freq=t.suggest_categorical("k_freq", [8, 16, 32]),
             d_cat=t.suggest_categorical("d_cat", [4, 8, 16]))
    return c


def _sg_leaf(t, ds):
    """leaf-one-hot -> FIXED head: only the leaf-feature kernel is tunable."""
    return dict(k_n_estimators=t.suggest_int("k_n_estimators", 100, 400, step=100),
                k_max_depth=t.suggest_int("k_max_depth", 3, 7))


def _sg_pca(t, ds):
    """PCA of leaf features -> FIXED head: tune #components (parallel to TKCE's
    embedding size) and the leaf-feature kernel."""
    return dict(n_components=t.suggest_categorical("n_components", [32, 64, 128]),
                k_n_estimators=t.suggest_int("k_n_estimators", 100, 400, step=100),
                k_max_depth=t.suggest_int("k_max_depth", 3, 7))


def _sg_tkce(t, ds):
    """TKCE search space: the ENCODER (embedding) and kernel are tuned; the
    downstream head is FIXED (FIXED_HEAD), so tuning shapes the representation."""
    enc_width = t.suggest_categorical("enc_width", [128, 256, 384, 512])
    enc_depth = t.suggest_int("enc_depth", 1, 3)
    return dict(
        # --- encoder (embedding) hyperparameters ---
        enc_hidden=tuple([enc_width] * enc_depth),      # encoder width x depth
        enc_dropout=t.suggest_float("enc_dropout", 0.0, 0.4),
        embedding_dim=t.suggest_categorical("embedding_dim", [32, 64, 128]),
        enc_lr=t.suggest_float("enc_lr", 3e-4, 3e-3, log=True),
        # --- contrastive / kernel hyperparameters ---
        temperature=t.suggest_float("temperature", 0.05, 0.5, log=True),
        pos_threshold=t.suggest_float("pos_threshold", 0.4, 0.7),
        pretrain_epochs=t.suggest_int("pretrain_epochs", 15, 40),
        k_n_estimators=t.suggest_int("k_n_estimators", 100, 400, step=100),
        k_max_depth=t.suggest_int("k_max_depth", 3, 7))


def _sg_tkce_joint(t, ds):
    c = _sg_tkce(t, ds)
    c.update(lambda_contrast=t.suggest_float("lambda_contrast", 0.05, 3.0, log=True))
    return c


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #
MODEL_SPECS = {
    "xgboost":        dict(suggest=_sg_xgboost, run=_run_tree("xgboost")),
    "lightgbm":       dict(suggest=_sg_lightgbm, run=_run_tree("lightgbm")),
    "catboost":       dict(suggest=_sg_catboost, run=_run_tree("catboost")),
    "random_forest":  dict(suggest=_sg_rf, run=_run_tree("random_forest")),
    "mlp_raw":        dict(suggest=_sg_fixed, run=_run_nn_raw("mlp")),
    "tabresnet_raw":  dict(suggest=_sg_fixed, run=_run_nn_raw("tabresnet")),
    "ft_transformer": dict(suggest=_sg_ft, run=_run_special("ft_transformer")),
    "num_embed_mlp":  dict(suggest=_sg_numembed, run=_run_special("num_embed_mlp")),
    "leafonehot_mlp": dict(suggest=_sg_leaf, run=_run_leaf_onehot),
    "pca_gbt_mlp":       dict(suggest=_sg_pca, run=_run_pca("mlp")),
    "pca_gbt_tabresnet": dict(suggest=_sg_pca, run=_run_pca("tabresnet")),
    "tkce2s_gbt_mlp":       dict(suggest=_sg_tkce, run=_run_tkce_twostage("gbt", "mlp")),
    "tkce2s_gbt_tabresnet": dict(suggest=_sg_tkce, run=_run_tkce_twostage("gbt", "tabresnet")),
    "tkce2s_mondrian_mlp":  dict(suggest=_sg_tkce, run=_run_tkce_twostage("mondrian", "mlp")),
    "tkce_joint_gbt_mlp":       dict(suggest=_sg_tkce_joint, run=_run_tkce_joint("gbt", "mlp")),
    "tkce_joint_gbt_tabresnet": dict(suggest=_sg_tkce_joint, run=_run_tkce_joint("gbt", "tabresnet")),
}

# Reasonable default model set for the subset run.
DEFAULT_MODELS = list(MODEL_SPECS.keys())


def tune_model(name, ds, n_trials, device, seed=0, timeout=None):
    """Return (best_cfg, best_val, study). Maximizes validation primary metric."""
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    spec = MODEL_SPECS[name]

    def objective(trial):
        cfg = spec["suggest"](trial, ds)
        cfg["device"] = str(device)
        try:
            out = spec["run"](cfg, ds, device, seed)
        except Exception as e:  # noqa: BLE001 - a bad config shouldn't kill the study
            raise optuna.TrialPruned() from e
        v = out["val"]
        return v if np.isfinite(v) else -1e9

    # A model with an empty search space (fixed head on raw features) has nothing
    # to tune — run it once instead of n_trials identical evaluations.
    try:
        if not spec["suggest"](optuna.trial.FixedTrial({}), ds):
            n_trials = 1
    except Exception:  # noqa: BLE001 - suggest needs params -> real search space
        pass

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials, timeout=timeout,
                   show_progress_bar=False)
    # Rebuild the full cfg dict from best_params via a fixed trial replay.
    best_cfg = _cfg_from_params(name, ds, study.best_trial.params)
    best_cfg["device"] = str(device)
    return best_cfg, study.best_value, study


def _cfg_from_params(name, ds, params):
    """Reconstruct the cfg dict the runner expects from Optuna's flat params."""
    import optuna
    fixed = optuna.trial.FixedTrial(params)
    return MODEL_SPECS[name]["suggest"](fixed, ds)


def run_model(name, cfg, ds, device, seed):
    """Evaluate a fixed cfg on a dataset/seed -> {'val':..., 'test':{...}}."""
    return MODEL_SPECS[name]["run"]({**cfg, "device": str(device)}, ds, device, seed)
