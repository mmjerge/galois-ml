"""Grokking study on finite-field multiplication.

Task: learn (x, y) -> x * y in the unit group GF(p^n)*, presented as
opaque element identities (the network never sees field coordinates).
GF(p^n)* is cyclic of order p^n - 1, so the task is secretly modular
addition of discrete logarithms - but the isomorphism is hidden.

Following Nanda et al. (ICLR 2023), who reverse-engineered grokked
modular-addition networks into a sparse-Fourier algorithm, we track:

* train/test accuracy (the grokking curve),
* Fourier concentration of the learned embeddings when rows are ordered
  by discrete log - sparse spectra mean the network found the hidden
  cyclic structure,
* a Galois-specific probe with no analogue in plain modular addition:
  whether the Frobenius map x -> x^p (multiplication of discrete logs
  by p, hence a permutation of Fourier frequencies) becomes an
  approximate *orthogonal* transform of embedding space, measured by
  Procrustes residual against random-permutation baselines.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from galois_ml.finite_field import GF


@dataclass
class GFMultiplicationTask:
    """Multiplication in GF(p^n)* with opaque element ids."""

    p: int
    n: int
    seed: int = 0

    def __post_init__(self) -> None:
        self.field = GF(self.p, self.n, seed=self.seed)
        self.q = self.field.q
        self.group_order = self.q - 1
        # opaque ids: enumerate nonzero elements in from_int order
        self.elements = [
            self.field.from_int(k) for k in range(1, self.q)
        ]
        self._id_of = {tuple(e): i for i, e in enumerate(self.elements)}
        self.dlog, self.generator_id = self._discrete_log_table()
        self.frobenius_perm = self._frobenius_permutation()

    def _discrete_log_table(self) -> tuple[np.ndarray, int]:
        """Find a generator g of GF(q)* and return dlog[i] = k with
        elements[i] = g^k, plus the id of g."""
        for gid, g in enumerate(self.elements):
            dlog = np.full(self.group_order, -1, dtype=np.int64)
            x = self.field.one()
            ok = True
            for k in range(self.group_order):
                i = self._id_of[tuple(x)]
                if dlog[i] != -1:  # order of g < q - 1: not a generator
                    ok = False
                    break
                dlog[i] = k
                x = self.field.mul(x, g)
            if ok:
                return dlog, gid
        raise RuntimeError("no generator found (impossible for a field)")

    def _frobenius_permutation(self) -> np.ndarray:
        """perm[i] = id of elements[i]^p."""
        return np.array(
            [
                self._id_of[tuple(self.field.frobenius(e))]
                for e in self.elements
            ],
            dtype=np.int64,
        )

    def product_id(self, i: int, j: int) -> int:
        prod = self.field.mul(self.elements[i], self.elements[j])
        return self._id_of[tuple(prod)]

    def all_pairs(self) -> tuple[torch.Tensor, torch.Tensor]:
        """All N^2 multiplication problems: X (N^2, 2) ids, y (N^2,)."""
        m = self.group_order
        pairs, labels = [], []
        for i in range(m):
            for j in range(m):
                pairs.append((i, j))
                labels.append(self.product_id(i, j))
        return torch.tensor(pairs), torch.tensor(labels)

    def split(
        self, train_fraction: float, seed: int = 0
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        X, y = self.all_pairs()
        rng = np.random.default_rng(seed)
        perm = rng.permutation(len(y))
        cut = int(train_fraction * len(y))
        tr, te = perm[:cut], perm[cut:]
        return X[tr], y[tr], X[te], y[te]


class GrokkingMLP(nn.Module):
    """Embedding pair -> MLP -> product id logits. Small on purpose:
    the point is analyzing what it learns, not capacity."""

    def __init__(self, n_elements: int, d_embed: int = 96, d_hidden: int = 384):
        super().__init__()
        self.embed = nn.Embedding(n_elements, d_embed)
        self.net = nn.Sequential(
            nn.Linear(2 * d_embed, d_hidden),
            nn.ReLU(),
            nn.Linear(d_hidden, n_elements),
        )

    def forward(self, pairs: torch.Tensor) -> torch.Tensor:
        e = self.embed(pairs)  # (B, 2, d)
        return self.net(e.flatten(1))


# -- interpretability probes ----------------------------------------------

def fourier_concentration(
    embeddings: np.ndarray, dlog: np.ndarray, top_frequencies: int = 5
) -> float:
    """Fraction of (non-DC) Fourier power carried by the strongest
    `top_frequencies` frequencies, with embedding rows ordered by
    discrete log.

    If the network has discovered the cyclic structure of GF(q)*, its
    embedding columns are near-sinusoidal functions of the discrete log
    and this concentration approaches 1. A random matrix gives roughly
    top_frequencies / (n_frequencies) - the flat-spectrum baseline.
    """
    order = np.argsort(dlog)  # row k of E_ordered is embedding of g^k
    E = embeddings[order] - embeddings[order].mean(axis=0, keepdims=True)
    spectrum = np.abs(np.fft.rfft(E, axis=0)) ** 2  # (n_freq, d)
    power = spectrum[1:].sum(axis=1)  # drop DC; total power per frequency
    total = power.sum()
    if total == 0:
        return 0.0
    top = np.sort(power)[::-1][:top_frequencies]
    return float(top.sum() / total)


def frobenius_procrustes_residual(
    embeddings: np.ndarray, permutation: np.ndarray
) -> float:
    """Relative residual of the best *orthogonal* map R aligning E to
    its row-permutation E[perm]: min_R ||E[perm] - E R||_F / ||E[perm]||_F.

    For a Fourier-structured embedding, the Frobenius permutation
    (dlog k -> p*k) only permutes/mixes frequency components, which an
    orthogonal map can realize, so the residual is small. For a generic
    permutation of a generic embedding it stays large.
    """
    from scipy.linalg import orthogonal_procrustes

    E = embeddings - embeddings.mean(axis=0, keepdims=True)
    target = E[permutation]
    R, _ = orthogonal_procrustes(E, target)
    resid = np.linalg.norm(target - E @ R) / np.linalg.norm(target)
    return float(resid)


def frobenius_alignment_zscore(
    embeddings: np.ndarray,
    frobenius_perm: np.ndarray,
    n_baseline: int = 50,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Compare the Frobenius Procrustes residual against the residual
    distribution of random permutations.

    Returns (frobenius_residual, baseline_mean, z_score); a strongly
    negative z-score means the Frobenius map is realized orthogonally
    in a way random permutations are not - i.e. the network's geometry
    respects the Galois action.
    """
    rng = np.random.default_rng(seed)
    frob = frobenius_procrustes_residual(embeddings, frobenius_perm)
    baseline = np.array([
        frobenius_procrustes_residual(
            embeddings, rng.permutation(len(embeddings))
        )
        for _ in range(n_baseline)
    ])
    std = float(baseline.std()) or 1e-12
    return frob, float(baseline.mean()), (frob - float(baseline.mean())) / std
