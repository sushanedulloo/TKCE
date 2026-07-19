"""Turn a tree kernel into training pairs for the contrastive stage.

Scalable design (works for tens of thousands of rows):

  * SampledPositiveIndex : instead of the O(n^2) full kernel, we sample a large
    pool of random candidate pairs, compute K only for those (O(pairs*T), chunked),
    and keep the ones with K >= pos_threshold, grouped by anchor. Feeds InfoNCE.
  * SoftPairDataset : random (i, j) pairs carrying the exact kernel value K(i,j)
    as a soft regression target. Feeds kernel_regression.

Tabular rows have no temporal order, so (unlike the stock version) there is no
time-gap guard; we simply drop self-pairs.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import Dataset


def _kernel_pairs_chunked(kernel, leaves, i, j, chunk=200_000):
    """K(i,j) for aligned index arrays, computed in chunks to bound memory."""
    out = np.empty(len(i), dtype=np.float32)
    for s in range(0, len(i), chunk):
        e = min(s + chunk, len(i))
        out[s:e] = kernel.kernel_pairs(leaves, i[s:e], j[s:e])
    return out


class SampledPositiveIndex:
    """Anchor -> positive-neighbour lists mined from sampled candidate pairs."""

    def __init__(self, kernel, leaves, pos_threshold=0.6, max_pos=50,
                 n_candidate_pairs=None, seed=0):
        n = leaves.shape[0]
        self.n = n
        rng = np.random.default_rng(seed)
        if n_candidate_pairs is None:
            n_candidate_pairs = min(50 * n, 2_000_000)
        ii = rng.integers(0, n, size=n_candidate_pairs)
        jj = rng.integers(0, n, size=n_candidate_pairs)
        keep = ii != jj
        ii, jj = ii[keep], jj[keep]
        K = _kernel_pairs_chunked(kernel, leaves, ii, jj)
        pos = K >= pos_threshold
        ii, jj = ii[pos], jj[pos]

        buckets = defaultdict(list)
        for a, b in zip(ii.tolist(), jj.tolist()):
            buckets[a].append(b)
            buckets[b].append(a)  # symmetric
        self.pos_lists = []
        for i in range(n):
            lst = buckets.get(i, [])
            if len(lst) > max_pos:
                lst = rng.choice(lst, max_pos, replace=False).tolist()
            self.pos_lists.append(np.array(lst, dtype=np.int64))

    def has_positive(self, i):
        return len(self.pos_lists[i]) > 0

    def coverage(self):
        return float(np.mean([len(p) > 0 for p in self.pos_lists]))


class AnchorPositiveDataset(Dataset):
    def __init__(self, X, index: SampledPositiveIndex, seed=0):
        self.X = torch.from_numpy(np.ascontiguousarray(X))
        self.index = index
        self.valid = [i for i in range(index.n) if index.has_positive(i)]
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.valid)

    def __getitem__(self, k):
        i = self.valid[k]
        j = int(self.rng.choice(self.index.pos_lists[i]))
        return self.X[i], self.X[j]


class SoftPairDataset(Dataset):
    def __init__(self, X, kernel, leaves, n_pairs, seed=0):
        self.X = torch.from_numpy(np.ascontiguousarray(X))
        n = X.shape[0]
        rng = np.random.default_rng(seed)
        self.i = rng.integers(0, n, size=n_pairs)
        self.j = rng.integers(0, n, size=n_pairs)
        self.K = _kernel_pairs_chunked(kernel, leaves, self.i, self.j).astype(np.float32)

    def __len__(self):
        return len(self.i)

    def __getitem__(self, k):
        return self.X[self.i[k]], self.X[self.j[k]], torch.tensor(self.K[k])
