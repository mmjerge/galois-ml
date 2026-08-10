"""Galois group criteria from zeta coefficients.

Two classifiers, both interpretable:

* `TheoremClassifier` - the deterministic criteria reverse-engineered
  from ML models and *proved* in Lee & Lee (arXiv:2508.06670), applied
  first; where they cannot decide from the first N coefficients it falls
  back on Chebotarev density comparison.

* `ChebotarevBayesClassifier` - the exact Bayes rule for the synthetic
  data model of `galois_ml.data`: for an unramified prime p, the
  observable signature (a_{p^k})_{p^k <= N} is determined by the inertia
  degree f, whose distribution is the group's Chebotarev measure. The
  paper's deterministic criteria are special cases: an observation with
  zero density under a group excludes that group. Examples proved there:

  - Cor 3.3:  degree-l^2 extensions are cyclic iff a_{p^l}(K) = 0 for
    some p (only C_{l^2} has inert primes).
  - Cor 3.15 / 3.20: sextic C6 vs S3 and decic C10 vs D5 - the
    nonabelian groups have no order-|G| element, hence no inert primes.
  - Cor 3.24: abelian octic K is C8 iff a_{p^4}(K) = 0 for some p, and
    C2^3 iff a_{p^2}(K) != 0 for all p.
  - Prop 3.33: D4 and Q8 admit the same decomposition types, so only
    densities distinguish them - that task is intrinsically statistical.
"""

from __future__ import annotations

import math

import numpy as np

from galois_ml.groups import GROUPS, TASKS, GaloisGroup
from galois_ml.zeta import euler_coefficient, primes_up_to


def _a(row: np.ndarray, n: int) -> float:
    """a_n(K) from a feature row indexed a_1..a_N."""
    return row[n - 1]


class ChebotarevBayesClassifier:
    """Exact Bayes classifier for the unramified Chebotarev data model.

    For each prime p, the visible prime-power coefficients
    (a_p, a_{p^2}, ..., a_{p^K}) with p^K <= N are a deterministic
    function of the inertia degree f (Cor 2.2), so each prime yields a
    categorical observation whose class-conditional distribution is the
    group's Chebotarev measure, pushed through that function (distinct f
    can collide when too few powers are visible; the pushforward handles
    that automatically). Independence across primes makes the likelihood
    a product. With equal priors, predict the argmax group.

    This is the ceiling for any classifier on this synthetic data, and a
    transparent description of what the trees/regressions approximate.
    """

    def __init__(self, task: str, n_coeffs: int = 1000):
        self.task = task
        self.n_coeffs = n_coeffs
        self.groups: list[GaloisGroup] = [GROUPS[g] for g in TASKS[task]]
        self.primes = primes_up_to(n_coeffs)
        degree = self.groups[0].order
        # visible powers per prime: p, p^2, ..., p^K <= N
        self._max_power = {
            p: max(k for k in range(1, degree + 1) if p**k <= n_coeffs)
            for p in self.primes
        }
        # per prime: map from observable signature -> density, per group
        self._signature_density: list[dict[int, dict[tuple, float]]] = []
        for group in self.groups:
            per_prime: dict[int, dict[tuple, float]] = {}
            orders, probs = group.frobenius_orders
            for p in self.primes:
                K = self._max_power[p]
                dens: dict[tuple, float] = {}
                for f, prob in zip(orders, probs):
                    g = group.order // int(f)
                    sig = tuple(
                        euler_coefficient(k, int(f), g)
                        for k in range(1, K + 1)
                    )
                    dens[sig] = dens.get(sig, 0.0) + float(prob)
                per_prime[p] = dens
            self._signature_density.append(per_prime)

    def _signature(self, row: np.ndarray, p: int) -> tuple:
        return tuple(
            int(_a(row, p**k)) for k in range(1, self._max_power[p] + 1)
        )

    def log_likelihoods(self, row: np.ndarray) -> np.ndarray:
        lls = np.zeros(len(self.groups))
        for p in self.primes:
            sig = self._signature(row, p)
            densities = [
                self._signature_density[gi][p].get(sig, 0.0)
                for gi in range(len(self.groups))
            ]
            if not any(densities):
                # Signature impossible under every candidate: the prime
                # is ramified (outside the unramified model). It carries
                # no group information here - skip it. (Ramified-type
                # criteria like the paper's Prop 3.4-3.8 could be added
                # to exploit these primes.)
                continue
            for gi, density in enumerate(densities):
                lls[gi] += math.log(density) if density > 0 else -math.inf
        return lls

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.array(
            [int(np.argmax(self.log_likelihoods(row))) for row in np.atleast_2d(X)]
        )


class TheoremClassifier:
    """Apply the paper's proven deterministic criteria first; when the
    first N coefficients cannot decide (the distinguishing Frobenius
    class need not appear among small primes), fall back to Chebotarev
    density comparison over the still-possible groups.

    Predicts label indices consistent with `TASKS[task]` and
    `galois_ml.data.make_dataset`.
    """

    def __init__(self, task: str, n_coeffs: int = 1000):
        if task not in TASKS:
            raise ValueError(f"unknown task {task!r}")
        if not hasattr(self, f"_predict_{task}"):
            raise ValueError(
                f"no proven criteria implemented for task {task!r} "
                f"(held-out task?); use ChebotarevBayesClassifier instead"
            )
        self.task = task
        self.n_coeffs = n_coeffs
        self._bayes = ChebotarevBayesClassifier(task, n_coeffs)
        primes = self._bayes.primes
        self._p2 = [p for p in primes if p * p <= n_coeffs]
        self._p3 = [p for p in primes if p**3 <= n_coeffs]
        self._p4 = [p for p in primes if p**4 <= n_coeffs]
        self._p5 = [p for p in primes if p**5 <= n_coeffs]

    def predict(self, X: np.ndarray) -> np.ndarray:
        method = getattr(self, f"_predict_{self.task}")
        return np.array([method(row) for row in np.atleast_2d(X)])

    def _bayes_among(self, row: np.ndarray, candidates: list[int]) -> int:
        """Maximum-likelihood choice restricted to candidate labels."""
        lls = self._bayes.log_likelihoods(row)
        mask = np.full_like(lls, -np.inf)
        mask[candidates] = lls[candidates]
        return int(np.argmax(mask))

    # -- degree l^2: cyclic iff some a_{p^l} vanishes (Cor 3.3) ----------

    def _predict_quartic(self, row) -> int:
        # C4 (label 0) iff a_{p^2} = 0 for some p; else C2^2 (label 1).
        return 0 if any(_a(row, p * p) == 0 for p in self._p2) else 1

    def _predict_nonic(self, row) -> int:
        # C9 (label 0) iff a_{p^3} = 0 for some p (Cor 3.3, l = 3).
        if any(_a(row, p**3) == 0 for p in self._p3):
            return 0
        return self._bayes_among(row, [0, 1])

    # -- sextic C6 vs S3 (Cor 3.14/3.15) ---------------------------------

    def _predict_sextic(self, row) -> int:
        # Inert prime (f = 6): a_{p^2} = a_{p^3} = 0. Impossible for S3.
        for p in self._p3:
            if _a(row, p * p) == 0 and _a(row, p**3) == 0:
                return 0  # C6
        return self._bayes_among(row, [0, 1])

    # -- decic C10 vs D5 (Cor 3.20) ---------------------------------------

    def _predict_decic(self, row) -> int:
        # Inert prime (f = 10): a_{p^2} = a_{p^5} = 0. Impossible for D5.
        for p in self._p5:
            if _a(row, p * p) == 0 and _a(row, p**5) == 0:
                return 0  # C10
        return self._bayes_among(row, [0, 1])

    # -- octic abelian: C8 vs C4xC2 vs C2^3 (Cor 3.24) --------------------

    def _predict_octic_abelian(self, row) -> int:
        # f = 8 (inert): a_{p^4} = 0. Only C8 has order-8 elements.
        if any(_a(row, p**4) == 0 for p in self._p4):
            return 0  # C8
        # f >= 4 leaves the signature a_{p^2} = 0; impossible for C2^3.
        f4_or_more = [_a(row, p * p) == 0 for p in self._p2]
        if not any(f4_or_more):
            return 2  # C2^3
        return self._bayes_among(row, [0, 1])

    # -- octic nonabelian: D4 vs Q8 (Prop 3.33: same types, densities only)

    def _predict_octic_nonabelian(self, row) -> int:
        return self._bayes_among(row, [0, 1])
