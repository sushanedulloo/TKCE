"""Tree kernels: the source of "what is similar" for the contrastive stage.

Every kernel exposes the same interface:

    kern = SomeKernel.fit(X_train, y_train, task_type, cfg)
    L    = kern.leaves(X)                    # (n, T) int leaf-id matrix
    K    = kern.kernel_pairs(L, i, j)        # (m,)  kernel value for aligned pairs
    Kmat = kern.kernel_matrix(La, Lb)        # (na, nb) full matrix (eval only)

Leaf co-occupancy kernel:  K(x_i, x_j) = (1/T) * sum_t 1[leaf_t(x_i) == leaf_t(x_j)]
which approximates the Laplace kernel exp(-lambda ||x_i - x_j||_1) (Balog 2016).

Three sources:
  * GBTKernel      - supervised gradient boosting (XGBoost). Target-led: encodes
                     the tree bias RELEVANT to the task. Primary method.
  * RFKernel       - Random Forest leaf co-occupancy. Supervised but bagged.
  * MondrianKernel - UNSUPERVISED random axis-aligned partitions (no labels).
                     The label-free "does a pure tree prior help?" ablation.
"""

from __future__ import annotations

import numpy as np


class _LeafKernel:
    """Shared leaf-co-occupancy math; subclasses only provide `leaves`."""

    def leaves(self, X: np.ndarray) -> np.ndarray:  # pragma: no cover - abstract
        raise NotImplementedError

    def kernel_pairs(self, leaves: np.ndarray, i: np.ndarray, j: np.ndarray) -> np.ndarray:
        agree = (leaves[i] == leaves[j]).astype(np.float32)   # (m, T)
        return agree.mean(axis=1)

    def kernel_matrix(self, leaves_a: np.ndarray, leaves_b: np.ndarray | None = None,
                      block: int = 512) -> np.ndarray:
        if leaves_b is None:
            leaves_b = leaves_a
        na, T = leaves_a.shape[0], leaves_a.shape[1]
        K = np.empty((na, leaves_b.shape[0]), dtype=np.float32)
        for s in range(0, na, block):
            e = min(s + block, na)
            eq = (leaves_a[s:e, None, :] == leaves_b[None, :, :]).astype(np.float32)
            K[s:e] = eq.mean(axis=2)
        return K


class GBTKernel(_LeafKernel):
    def __init__(self, model):
        self.model = model

    @classmethod
    def fit(cls, X, y, task_type, cfg):
        import xgboost as xgb
        params = dict(
            n_estimators=cfg.get("n_estimators", 200),
            max_depth=cfg.get("max_depth", 4),
            learning_rate=cfg.get("learning_rate", 0.1),
            subsample=cfg.get("subsample", 0.8),
            colsample_bytree=cfg.get("colsample_bytree", 0.8),
            random_state=cfg.get("random_state", 42),
            n_jobs=-1, tree_method="hist",
        )
        if task_type == "classification":
            model = xgb.XGBClassifier(**params, eval_metric="logloss")
        else:
            model = xgb.XGBRegressor(**params)
        model.fit(X, y)
        return cls(model)

    def leaves(self, X):
        return self.model.apply(X).astype(np.int32)


class RFKernel(_LeafKernel):
    def __init__(self, model):
        self.model = model

    @classmethod
    def fit(cls, X, y, task_type, cfg):
        from sklearn.ensemble import (RandomForestClassifier,
                                      RandomForestRegressor)
        params = dict(
            n_estimators=cfg.get("n_estimators", 200),
            max_depth=cfg.get("max_depth", None),
            min_samples_leaf=cfg.get("min_samples_leaf", 5),
            n_jobs=-1, random_state=cfg.get("random_state", 42),
        )
        Model = RandomForestClassifier if task_type == "classification" \
            else RandomForestRegressor
        model = Model(**params).fit(X, y)
        return cls(model)

    def leaves(self, X):
        return self.model.apply(X).astype(np.int32)


class MondrianKernel(_LeafKernel):
    """Unsupervised random axis-aligned partitions (Mondrian-style).

    For each of T trees we recursively split the bounding box on a random
    feature at a random threshold, up to a max depth or a lifetime budget.
    No labels are used, so the resulting kernel is a pure, task-agnostic
    smoothness prior over the feature space.
    """

    def __init__(self, trees, lo, hi):
        self.trees = trees          # list of split-node lists
        self.lo = lo                # per-feature train min (for bounding box)
        self.hi = hi

    @classmethod
    def fit(cls, X, y, task_type, cfg):
        rng = np.random.default_rng(cfg.get("random_state", 42))
        T = cfg.get("n_estimators", 200)
        max_depth = cfg.get("max_depth", 6)
        X = np.asarray(X, dtype=np.float64)
        lo, hi = X.min(axis=0), X.max(axis=0)
        span = np.maximum(hi - lo, 1e-9)
        d = X.shape[1]
        trees = []
        for _ in range(T):
            # Each tree is a flat list of (feature, threshold, left_id, right_id);
            # leaves are encoded as negative ids.
            nodes = []

            def build(depth, box_lo, box_hi):
                # Probability of splitting decreases with depth; stop at max_depth.
                widths = box_hi - box_lo
                if depth >= max_depth or widths.sum() <= 1e-9:
                    leaf_id = -(len(nodes) + 1)
                    return leaf_id
                # Sample split feature proportional to box width (Mondrian rule).
                p = widths / widths.sum()
                f = rng.choice(d, p=p)
                thr = rng.uniform(box_lo[f], box_hi[f])
                node_idx = len(nodes)
                nodes.append([f, thr, None, None])
                lo_l, hi_l = box_lo.copy(), box_hi.copy(); hi_l[f] = thr
                lo_r, hi_r = box_lo.copy(), box_hi.copy(); lo_r[f] = thr
                nodes[node_idx][2] = build(depth + 1, lo_l, hi_l)
                nodes[node_idx][3] = build(depth + 1, lo_r, hi_r)
                return node_idx

            root = build(0, lo.copy(), hi.copy())
            trees.append((nodes, root))
        return cls(trees, lo, hi)

    def _leaf_of(self, tree, x):
        nodes, node = tree
        while node >= 0:
            f, thr, left, right = nodes[node]
            node = left if x[f] <= thr else right
        return -node  # positive leaf label

    def leaves(self, X):
        X = np.asarray(X, dtype=np.float64)
        out = np.empty((X.shape[0], len(self.trees)), dtype=np.int32)
        for t, tree in enumerate(self.trees):
            for i in range(X.shape[0]):
                out[i, t] = self._leaf_of(tree, X[i])
        return out


KERNELS = {"gbt": GBTKernel, "rf": RFKernel, "mondrian": MondrianKernel}


def build_kernel(name: str, X, y, task_type, cfg):
    return KERNELS[name].fit(X, y, task_type, cfg)
