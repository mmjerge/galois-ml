"""Synthetic zeta-coefficient datasets for Galois group classification.

Each sample is a "virtual number field": we fix a Galois group G, sample
the Frobenius order of every prime p <= N independently from the
Chebotarev distribution of G (see `galois_ml.groups`), and compute the
resulting zeta coefficients a_1, ..., a_N. This models the unramified
decomposition data of a random Galois extension with group G, which is
exactly the structural signal the models in Lee & Lee (arXiv:2508.06670)
were shown to exploit.

Unlike LMFDB data, samples here are exchangeable and the class balance is
controlled, which makes ablations clean. The trade-off is that ramified
primes and the arithmetic correlations of genuine fields are not modeled.
"""

from __future__ import annotations

import numpy as np

from galois_ml.groups import GROUPS, TASKS
from galois_ml.zeta import primes_up_to, zeta_coefficients


def make_dataset(
    task: str,
    n_samples: int,
    n_coeffs: int = 1000,
    seed: int = 0,
    balanced: bool = True,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Generate a zeta-coefficient classification dataset.

    Args:
        task: one of the keys of `galois_ml.groups.TASKS`
            ("quartic", "sextic", "octic_abelian", "octic_nonabelian",
            "nonic", "decic").
        n_samples: number of virtual fields to generate.
        n_coeffs: number of zeta coefficients per sample (features are
            a_1, ..., a_{n_coeffs}).
        seed: RNG seed.
        balanced: if True, classes are (nearly) equally represented;
            otherwise classes are drawn uniformly at random.

    Returns:
        (X, y, class_names) with X of shape (n_samples, n_coeffs),
        y[i] the index of the sample's group in class_names.
    """
    if task not in TASKS:
        raise ValueError(f"unknown task {task!r}; choose from {sorted(TASKS)}")
    groups = [GROUPS[name] for name in TASKS[task]]
    rng = np.random.default_rng(seed)
    primes = primes_up_to(n_coeffs)

    X = np.empty((n_samples, n_coeffs), dtype=np.float64)
    y = np.empty(n_samples, dtype=np.int64)
    for i in range(n_samples):
        gi = i % len(groups) if balanced else int(rng.integers(len(groups)))
        group = groups[gi]
        fs = group.sample_frobenius_orders(len(primes), rng)
        frob = {p: int(f) for p, f in zip(primes, fs)}
        X[i] = zeta_coefficients(n_coeffs, frob, group.order)
        y[i] = gi

    perm = rng.permutation(n_samples)
    return X[perm], y[perm], [g.name for g in groups]
