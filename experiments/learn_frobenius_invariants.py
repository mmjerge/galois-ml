"""Does building in Galois symmetry help a network learn arithmetic?

Task: classify whether x in GF(p^n)* is a square. This is a
Frobenius-invariant property (sigma(y^2) = sigma(y)^2), and in normal
coordinates the Frobenius acts by cyclic shift - so a Galois-equivariant
network (cyclic group convolutions + invariant pooling) matches the label
symmetry by construction, while a plain MLP must learn it from data.

We compare sample efficiency: accuracy on held-out field elements as a
function of training set size. This is the "inductive bias of arithmetic
symmetry" experiment in its simplest form.

Run:  python experiments/learn_frobenius_invariants.py
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from galois_ml.equivariant import GaloisInvariantNet, PlainMLP
from galois_ml.finite_field import GF

P, N = 5, 4  # GF(625); Galois group C_4
TRAIN_SIZES = [50, 100, 200, 400]
SEEDS = [0, 1, 2]
EPOCHS = 300


def build_dataset() -> tuple[torch.Tensor, torch.Tensor]:
    """All nonzero elements of GF(p^n) in normal coordinates, labeled by
    squareness."""
    field = GF(P, N, seed=0)
    nb = field.normal_basis(seed=0)
    coords, labels = [], []
    for x in field.all_elements():
        if not x.any():
            continue
        coords.append(nb.to_normal(x))
        labels.append(int(field.is_square(x)))
    X = torch.tensor(np.array(coords), dtype=torch.long)
    y = torch.tensor(labels, dtype=torch.long)
    # sanity: the label really is Frobenius-invariant, i.e. constant on
    # orbits (cyclic shifts in normal coordinates)
    lookup = {tuple(c.tolist()): int(t) for c, t in zip(X, y)}
    for c, t in zip(X, y):
        shifted = tuple(np.roll(c.numpy(), 1).tolist())
        assert lookup[shifted] == int(t), "label not Galois-invariant?!"
    return X, y


def train(model: nn.Module, X: torch.Tensor, y: torch.Tensor) -> None:
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()
    model.train()
    for _ in range(EPOCHS):
        opt.zero_grad()
        loss = loss_fn(model(X), y)
        loss.backward()
        opt.step()


@torch.no_grad()
def accuracy(model: nn.Module, X: torch.Tensor, y: torch.Tensor) -> float:
    model.eval()
    return float((model(X).argmax(dim=1) == y).float().mean())


def main() -> None:
    X, y = build_dataset()
    print(f"GF({P}^{N}) = GF({P**N}); {len(X)} nonzero elements, "
          f"{int(y.sum())} squares\n")

    header = f"{'train size':>10} {'equivariant':>14} {'plain MLP':>12}"
    print(header)
    print("-" * len(header))
    for size in TRAIN_SIZES:
        eq_accs, mlp_accs = [], []
        for seed in SEEDS:
            gen = np.random.default_rng(seed)
            perm = gen.permutation(len(X))
            tr, te = perm[:size], perm[size:]
            torch.manual_seed(seed)
            eq = GaloisInvariantNet(P, N, hidden_channels=32, depth=3)
            mlp = PlainMLP(P, N, hidden=64, depth=3)
            train(eq, X[tr], y[tr])
            train(mlp, X[tr], y[tr])
            eq_accs.append(accuracy(eq, X[te], y[te]))
            mlp_accs.append(accuracy(mlp, X[te], y[te]))
        print(
            f"{size:>10} "
            f"{np.mean(eq_accs):>8.4f} +/- {np.std(eq_accs):.3f} "
            f"{np.mean(mlp_accs):>8.4f} +/- {np.std(mlp_accs):.3f}"
        )

    n_eq = sum(p.numel() for p in GaloisInvariantNet(P, N).parameters())
    n_mlp = sum(p.numel() for p in PlainMLP(P, N, hidden=64).parameters())
    print(f"\nparameters: equivariant {n_eq}, MLP {n_mlp}")


if __name__ == "__main__":
    main()
