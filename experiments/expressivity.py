"""Expressivity of Galois-equivariant networks: what they can and
cannot represent.

An invariant architecture trades capacity for bias in a sharp, provable
way: a Frobenius-invariant network is constant on Galois orbits, so

* it can in principle fit ANY invariant labeling (one value per orbit),
* it can NEVER fit a non-invariant labeling beyond the "orbit majority"
  ceiling: the best accuracy achievable by any function constant on
  orbits.

We verify both empirically on GF(5^4): train on random labelings that
are (a) constant on Frobenius orbits and (b) unconstrained, and compare
training accuracy against the computable ceiling. The plain MLP has no
such ceiling and memorizes both.

Run:  python experiments/expressivity.py
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from galois_ml.equivariant import GaloisInvariantNet, PlainMLP
from galois_ml.finite_field import GF

P, N = 5, 4
EPOCHS = 2000
SEED = 0


def frobenius_orbits(coords: np.ndarray) -> list[list[int]]:
    """Partition normal-coordinate vectors into Frobenius (cyclic shift)
    orbits; returns lists of row indices."""
    index = {tuple(c): i for i, c in enumerate(map(tuple, coords))}
    seen, orbits = set(), []
    for i, c in enumerate(coords):
        if i in seen:
            continue
        orbit = []
        x = np.array(c)
        for _ in range(N):
            j = index[tuple(x)]
            if j not in orbit:
                orbit.append(j)
            x = np.roll(x, 1)
        seen.update(orbit)
        orbits.append(orbit)
    return orbits


def orbit_majority_ceiling(orbits: list[list[int]], y: np.ndarray) -> float:
    """Best possible accuracy of any function constant on orbits."""
    correct = sum(
        max(np.sum(y[orbit] == label) for label in np.unique(y))
        for orbit in orbits
    )
    return correct / len(y)


def train_to_fit(model: nn.Module, X: torch.Tensor, y: torch.Tensor) -> float:
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    loss_fn = nn.CrossEntropyLoss()
    model.train()
    for _ in range(EPOCHS):
        opt.zero_grad()
        loss_fn(model(X), y).backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        return float((model(X).argmax(1) == y).float().mean())


def main() -> None:
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)

    field = GF(P, N, seed=0)
    nb = field.normal_basis(seed=0)
    coords = np.array(
        [nb.to_normal(x) for x in field.all_elements() if x.any()]
    )
    X = torch.tensor(coords, dtype=torch.long)
    orbits = frobenius_orbits(coords)
    print(f"GF({P**N}): {len(coords)} nonzero elements, "
          f"{len(orbits)} Frobenius orbits\n")

    # (a) random labels constant on orbits (invariant target)
    y_inv = np.empty(len(coords), dtype=np.int64)
    for orbit in orbits:
        y_inv[orbit] = rng.integers(2)
    # (b) unconstrained random labels (non-invariant target)
    y_rand = rng.integers(2, size=len(coords)).astype(np.int64)

    for label_name, y_np in [("invariant", y_inv), ("unconstrained", y_rand)]:
        y = torch.tensor(y_np)
        ceiling = orbit_majority_ceiling(orbits, y_np)
        eq = GaloisInvariantNet(P, N, hidden_channels=64, depth=3)
        mlp = PlainMLP(P, N, hidden=128, depth=3)
        eq_fit = train_to_fit(eq, X, y)
        mlp_fit = train_to_fit(mlp, X, y)
        print(f"target: random {label_name} labels")
        print(f"  orbit-majority ceiling (binding for invariant nets): "
              f"{ceiling:.4f}")
        print(f"  equivariant net fit:  {eq_fit:.4f}")
        print(f"  plain MLP fit:        {mlp_fit:.4f}\n")

    print("Expected: the equivariant net fits invariant labels (near 1.0)"
          "\nbut is capped at the ceiling on unconstrained labels, while"
          "\nthe MLP memorizes both. Capacity is traded for symmetry.")


if __name__ == "__main__":
    main()
