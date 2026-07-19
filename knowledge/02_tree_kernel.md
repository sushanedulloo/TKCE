# 02 · The tree kernel (and its Mondrian / Laplace roots)

## What a kernel is
A **kernel** `K(x, y)` is a similarity score between two points. Higher = more
similar. Many kernels can be written as a dot product of *feature maps*:
`K(x, y) = ⟨φ(x), φ(y)⟩`. That identity is the hook this whole project hangs on.

## Leaves as one-hot codes
A decision tree sends every point to exactly one **leaf**. Encode "which leaf"
as a one-hot vector `zᵗ` for tree `t` (a vector of 0s with a single 1).

Key property of one-hot vectors:
```
⟨zᵢᵗ, zⱼᵗ⟩ = 1  if xᵢ and xⱼ share the same leaf in tree t
           = 0  otherwise
```
Because the only way the dot product is non-zero is if both "1"s sit in the same
position — i.e. the same leaf. So "same box?" is literally a dot product.

## From one tree to a forest
One tree is noisy. A forest of `T` trees (XGBoost) is robust. Weight each tree
by `wₜ` and average:

```
K(xᵢ, xⱼ) = (1/T) · Σₜ  wₜ · ⟨zᵢᵗ, zⱼᵗ⟩
          = (weighted) fraction of trees in which the two points share a leaf
```

- `K = 1` → same leaf in every tree → very similar.
- `K = 0` → never share a leaf → very dissimilar.

**Implementation shortcut:** XGBoost's leaf-id matrix (`model.apply(X)`, shape
`(n, T)`) lets us compute `K` by elementwise integer comparison and averaging —
we never build the big one-hot matrices. See `src/tree_kernel.py`.

## Why this equals the Laplace kernel (Mondrian result)
Balog et al., *The Mondrian Kernel* (2016), show that random tree partitions
(a Mondrian process) give a **random-feature approximation** to the isotropic
**Laplace kernel**:

```
k(x, x′) = exp(−λ · ‖x − x′‖₁) = exp(−λ · Σ_d |x_d − x′_d|)
```

- `‖x − x′‖₁` — sum of absolute per-feature differences (Manhattan distance).
- `λ` ("lifetime") — how fast similarity decays with distance. Large `λ` = only
  very close points count as similar.

Reading of the formula: **the fraction of trees where two points share a leaf
behaves like `exp(−λ·distance)`** — near points almost always co-occupy leaves
(K≈1), far points rarely do (K≈0). XGBoost gives a *supervised, data-adaptive*
version of the same object; a Mondrian process gives the *random* version.

Practical upshot: the tree kernel is not arbitrary. It is a fast, learnable
stand-in for a smooth, classical similarity. That is what makes it a sound
target to train an embedding against.

## Supervised vs random partitions
- **Supervised (this repo's default):** XGBoost fit on a target (next-day
  direction / return). The kernel then reflects similarity *relevant to what you
  predict* — a "target-led" kernel (CatFormer notes' terminology).
- **Random (pure Mondrian):** partitions drawn without a target. General-purpose
  similarity, no labels required. Swap in by replacing the `TreeKernel.fit`
  source; the rest of the pipeline is unchanged.

## Two knobs that matter most
- `xgboost.max_depth` — deeper trees = smaller leaves = stricter similarity
  (fewer points share a leaf, kernel becomes sparse).
- `xgboost.n_estimators` (T) — more trees = smoother, more reliable kernel.

If almost no pairs clear `pairs.pos_threshold`, your leaves are too fine: reduce
`max_depth`, raise `n_estimators`, or lower the threshold.
