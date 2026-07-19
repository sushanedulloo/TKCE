# 04 · References & how each source maps onto the code

## How the sources map onto this pipeline

| Source | What we use from it | Where in code |
|---|---|---|
| **CatFormer notes** | Trees → one-hot leaf vectors → tree kernel `K = (1/T)Σ wₜ⟨zᵢᵗ,zⱼᵗ⟩`; Siamese `φ` trained so `⟨φ(xᵢ),φ(xⱼ)⟩ ≈ K`; the "data-driven, not explicitly specified" kernel. | `src/tree_kernel.py`, `src/model.py`, `src/losses.py` |
| **Balog et al., *The Mondrian Kernel* (2016)** | Leaf co-occupancy ≈ Laplace kernel `exp(−λ‖x−x′‖₁)`; random partitions as a fast, width-reusable feature map that justifies the tree kernel. | `src/tree_kernel.py` (docstrings), `knowledge/02` |
| **Weng, *Contrastive Representation Learning* (2021)** | The loss menu (contrastive, triplet, N-pair, InfoNCE, soft-nearest-neighbour); key ingredients — large batches, hard-negative mining; collapse avoidance. | `src/losses.py`, `src/pairs.py`, `src/train.py` |
| **Rusak et al., *AnInfoNCE* (arXiv 2407.00143)** | InfoNCE assumes equal variance across latent factors; the anisotropic generalization gives each factor its own learnable scale so dominant directions aren't collapsed. | `AnInfoNCELoss` in `src/losses.py` |
| **TUM Siamese-network lecture** | The twin-network / shared-weights architecture and pair-based distance learning underpinning `φ`. | `src/model.py` |

## Citations

- Balog, M., Lakshminarayanan, B., Ghahramani, Z., Roy, D. M., & Teh, Y. W.
  (2016). *The Mondrian Kernel*. UAI 2016. arXiv:1606.05241.
- Weng, L. (2021). *Contrastive Representation Learning*. Lil'Log.
  https://lilianweng.github.io/posts/2021-05-31-contrastive/
- *AnInfoNCE: Identifying the Latent Factors in Anisotropic Contrastive
  Learning*. arXiv:2407.00143.
- Chopra, S., Hadsell, R., & LeCun, Y. (2005). *Learning a similarity metric
  discriminatively, with application to face verification*. CVPR.
- Schroff, F., Kalenichenko, D., & Philbin, J. (2015). *FaceNet* (triplet loss).
  CVPR. arXiv:1503.03832.
- van den Oord, A., Li, Y., & Vinyals, O. (2018). *Representation Learning with
  Contrastive Predictive Coding* (InfoNCE). arXiv:1807.03748.
- Chen, T., Kornblith, S., Norouzi, M., & Hinton, G. (2020). *SimCLR*.
  arXiv:2002.05709.
- Rahimi, A., & Recht, B. (2007). *Random Features for Large-Scale Kernel
  Machines*. NeurIPS. (Random-feature background for the Mondrian kernel.)

## Suggested reading order
1. `01_overview.md` — the whole idea in one page.
2. `02_tree_kernel.md` — where the similarity signal comes from.
3. `03_contrastive_learning.md` — how similarity becomes geometry.
4. This file — sources and exact code mapping.
