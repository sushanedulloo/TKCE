# 03 · Contrastive learning & the Siamese encoder

## Goal
Learn an encoder `φ` that maps each row to a vector so that:
- similar points (high tree kernel `K`) → **close** embeddings,
- dissimilar points (low `K`) → **far** embeddings.

Formally we want `⟨φ(xᵢ), φ(xⱼ)⟩ ≈ K(xᵢ, xⱼ)`.

## The Siamese ("twin") network
We use **one** encoder `φ` and apply it to **both** members of a pair with the
**same weights**. That weight sharing is what "Siamese" means.

```
xᵢ ─►[ φ ]─► φ(xᵢ) ┐
                    ├─► compare (dot product / distance)
xⱼ ─►[ φ ]─► φ(xⱼ) ┘   (same φ, shared weights)
```

`φ` here is a small MLP. We **L2-normalize** its output so every embedding lies
on the unit sphere; then the dot product equals cosine similarity in `[-1, 1]`,
which lines up with the kernel's `[0, 1]` range. (See `src/model.py`.)

Linear `φ(x) = Wx` gives a linear embedding; stacking layers with nonlinearities
`σ(W₂ σ(W₁x + b₁) + b₂)` gives a non-linear one (more expressive). Start linear
to sanity-check, then go deeper.

## Positive and negative pairs
Contrastive learning needs pairs:
- **Positive** `(xᵢ, xⱼ⁺)`: should be close — drawn from high `K`.
- **Negative** `(xᵢ, xⱼ⁻)`: should be far — drawn from low `K` (in-batch for
  InfoNCE).

**Hard negatives** (low `K` but currently close in embedding space) are the most
informative; large batches expose more of them. **Time-gap** guarding avoids
pairing near-identical adjacent days. (See `src/pairs.py`.)

## The loss menu (all in `src/losses.py`)

### 1. Kernel regression (matches the CatFormer objective exactly)
```
ℓ = ( ⟨φ(xᵢ), φ(xⱼ)⟩ − K(xᵢ, xⱼ) )²
```
Directly forces the dot product to equal the kernel. Simplest starting point.

### 2. Contrastive loss (Hadsell/Chopra)
With `D = ‖φ(xᵢ) − φ(xⱼ)‖` and margin `ε`:
```
ℓ = 1[similar]·D²  +  1[dissimilar]·max(0, ε − D)²
```
Similar pairs: shrink `D`. Dissimilar pairs: push apart up to margin `ε`.

### 3. Triplet loss
Anchor `x`, positive `x⁺`, negative `x⁻`:
```
ℓ = max(0, ‖φ(x)−φ(x⁺)‖² − ‖φ(x)−φ(x⁻)‖² + ε)
```
Make the positive closer than the negative by at least `ε`.

### 4. InfoNCE (SimCLR / CLIP style) — usually the strongest
One positive vs many in-batch negatives, temperature `τ`:
```
ℓ = − log  exp(sim(x, x⁺)/τ) / Σ_all exp(sim(x, ·)/τ)
```
Small `τ` sharpens focus on the hardest negatives.

### 5. AnInfoNCE (anisotropic InfoNCE, arXiv 2407.00143)
Plain InfoNCE implicitly treats every embedding dimension as equally important.
In market data a few latent factors (e.g. volatility regime) dominate, and plain
InfoNCE can **collapse** the rest. AnInfoNCE uses a **learnable positive scale
`s_d` per dimension**:
```
sim(a, b) = Σ_d  s_d · a_d · b_d ,   s_d = softplus(θ_d) > 0
```
so the model stretches important directions and shrinks noisy ones instead of
discarding them. Use it when plain InfoNCE squashes everything into a narrow band.

## Avoiding collapse
Failure mode: `φ` maps everything to one vector (all pairs look "similar", loss
looks low, embeddings useless). Guards used here:
- keep **negatives** in the loss (contrastive/triplet/InfoNCE all do),
- **L2-normalize** outputs,
- **BatchNorm** (empirically injects an implicit contrastive effect).

## How to read success
After training, check `kernel_reconstruction.png`: embedding cosine similarity
should increase with the tree kernel `K` (high Pearson/Spearman). If it does, the
encoder learned the tree's notion of similarity. If it is flat, raise epochs,
lower `τ`, or widen the positive/negative gap in `config.yaml`.
