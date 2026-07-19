# 01 · Overview: from a table to an embedding space

## The problem
You have a table — rows are examples (e.g. one trading day), columns are
features (open, close, volume, moving averages, RSI, ...). You want a **vector
embedding** for each row such that **similar rows have similar vectors**.

The hard part is not the network. It is **defining "similar."** For images you
can crop or rotate to make a matching pair. For a stock row there is no obvious
augmentation, and no ready-made label saying "these two days are alike."

## The idea in one sentence
> Let a trained **XGBoost forest** define similarity for free — two points are
> similar if they land in the **same leaf** across many trees — then train a
> **Siamese network** with **contrastive learning** so that similarity becomes
> geometry (close vectors = similar points).

## The three stages

1. **Tree kernel (similarity source).**
   Train XGBoost. For each tree, note which leaf a point lands in. Two points
   sharing a leaf score 1, else 0. Average over all `T` trees:
   `K(xᵢ, xⱼ) = (1/T)·Σₜ wₜ·1[same leaf]`. This number in `[0,1]` is the
   similarity. It is *data-driven*, not hand-written. (See `02_tree_kernel.md`.)

2. **Pairs.**
   Turn `K` into training signal: positive pairs (high `K`), negative pairs
   (low `K`). Or keep `K` as a soft regression target.

3. **Contrastive embedding.**
   Train a Siamese encoder `φ` so `⟨φ(xᵢ), φ(xⱼ)⟩ ≈ K(xᵢ, xⱼ)`: pull similar
   pairs together, push dissimilar pairs apart. `φ(x)` is your embedding.
   (See `03_contrastive_learning.md`.)

## Why this is principled, not a hack
The Mondrian-kernel result says leaf co-occupancy approximates the Laplace
kernel `exp(−λ‖x−x′‖₁)`. So "fraction of trees where two points share a leaf"
≈ a smooth, well-understood similarity that decays with distance. Trees are
just a fast, data-adaptive way to compute it. (See `02_tree_kernel.md`.)

## What you can do with the embeddings
- Cluster days into **market regimes** (calm vs turbulent).
- Retrieve **"days like today"** by nearest neighbour in embedding space.
- Feed embeddings into a small head for **downstream prediction**.
- Detect anomalies (days far from everything else).

## Mental model
- **Trees answer:** *what is similar?* (a kernel)
- **Contrastive learning answers:** *how do I lay that out as geometry?*
- **The Siamese net is the bridge** that transfers similarity from "trees" into
  "distances in a vector space."
