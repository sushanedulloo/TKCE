"""Tree-Kernel Contrastive Embeddings (TKCE) — tabular representation learning.

Pipeline: tree kernel (GBT / Mondrian / RF) -> Siamese contrastive encoder ->
downstream MLP / TabResNet, compared against tree and NN baselines on the
Grinsztajn (2022) tabular benchmark.
"""
