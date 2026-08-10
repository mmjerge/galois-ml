"""Galois groups appearing in Lee & Lee (arXiv:2508.06670).

For classifying Galois groups from zeta coefficients, the only group data
we need for the *unramified* primes is the distribution of Frobenius
element orders: by the Chebotarev density theorem, the Frobenius conjugacy
class of an unramified prime p is equidistributed with respect to the
counting measure |C|/|G|, and the inertia degree f of p equals the order
of its Frobenius element. The decomposition type is then (e, f, g) =
(1, f, |G|/f).

Each group is therefore recorded by its multiset of element orders.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class GaloisGroup:
    """A finite group, viewed as a candidate Galois group Gal(K/Q)."""

    name: str
    order: int
    element_orders: dict[int, int]  # element order -> number of elements
    is_abelian: bool
    is_cyclic: bool

    def __post_init__(self) -> None:
        total = sum(self.element_orders.values())
        if total != self.order:
            raise ValueError(
                f"{self.name}: element order counts sum to {total}, "
                f"expected {self.order}"
            )

    @property
    def frobenius_orders(self) -> tuple[np.ndarray, np.ndarray]:
        """Possible Frobenius orders f and their Chebotarev densities."""
        orders = np.array(sorted(self.element_orders), dtype=np.int64)
        probs = np.array(
            [self.element_orders[int(o)] / self.order for o in orders]
        )
        return orders, probs

    def sample_frobenius_orders(
        self, size: int, rng: np.random.Generator
    ) -> np.ndarray:
        """Sample inertia degrees f for `size` unramified primes."""
        orders, probs = self.frobenius_orders
        return rng.choice(orders, size=size, p=probs)


def _g(name, order, element_orders, *, abelian, cyclic) -> GaloisGroup:
    return GaloisGroup(name, order, element_orders, abelian, cyclic)


#: All Galois groups studied in the paper, keyed by name.
GROUPS: dict[str, GaloisGroup] = {
    # Degree 4 (4T1, 4T2)
    "C4": _g("C4", 4, {1: 1, 2: 1, 4: 2}, abelian=True, cyclic=True),
    "C2^2": _g("C2^2", 4, {1: 1, 2: 3}, abelian=True, cyclic=False),
    # Degree 6 (6T1, 6T2)
    "C6": _g("C6", 6, {1: 1, 2: 1, 3: 2, 6: 2}, abelian=True, cyclic=True),
    "S3": _g("S3", 6, {1: 1, 2: 3, 3: 2}, abelian=False, cyclic=False),
    # Degree 8, abelian (8T1, 8T2, 8T3)
    "C8": _g("C8", 8, {1: 1, 2: 1, 4: 2, 8: 4}, abelian=True, cyclic=True),
    "C4xC2": _g("C4xC2", 8, {1: 1, 2: 3, 4: 4}, abelian=True, cyclic=False),
    "C2^3": _g("C2^3", 8, {1: 1, 2: 7}, abelian=True, cyclic=False),
    # Degree 8, nonabelian (8T4, 8T5)
    "D4": _g("D4", 8, {1: 1, 2: 5, 4: 2}, abelian=False, cyclic=False),
    "Q8": _g("Q8", 8, {1: 1, 2: 1, 4: 6}, abelian=False, cyclic=False),
    # Degree 9 (9T1, 9T2)
    "C9": _g("C9", 9, {1: 1, 3: 2, 9: 6}, abelian=True, cyclic=True),
    "C3^2": _g("C3^2", 9, {1: 1, 3: 8}, abelian=True, cyclic=False),
    # Degree 10 (10T1, 10T2)
    "C10": _g("C10", 10, {1: 1, 2: 1, 5: 4, 10: 4}, abelian=True, cyclic=True),
    "D5": _g("D5", 10, {1: 1, 2: 5, 5: 4}, abelian=False, cyclic=False),
    # ---- Held-out degrees (NOT analyzed in arXiv:2508.06670) ----
    # These give contamination-resistant benchmark variants: the same
    # generative model, but the distinguishing criteria appear in no
    # published paper. Element order counts: phi(d) elements of order d
    # for each divisor d in the cyclic case.
    # Degree 14 (14T1, 14T2)
    "C14": _g("C14", 14, {1: 1, 2: 1, 7: 6, 14: 6}, abelian=True, cyclic=True),
    "D7": _g("D7", 14, {1: 1, 2: 7, 7: 6}, abelian=False, cyclic=False),
    # Degree 21: cyclic vs the Frobenius group C7:C3 (14 elements of
    # order 3 in the two nontrivial cosets, 6 of order 7)
    "C21": _g("C21", 21, {1: 1, 3: 2, 7: 6, 21: 12}, abelian=True, cyclic=True),
    "C7:C3": _g("C7:C3", 21, {1: 1, 3: 14, 7: 6}, abelian=False, cyclic=False),
    # Degree 25 (l = 5 case of the degree-l^2 family)
    "C25": _g("C25", 25, {1: 1, 5: 4, 25: 20}, abelian=True, cyclic=True),
    "C5^2": _g("C5^2", 25, {1: 1, 5: 24}, abelian=True, cyclic=False),
}

#: Classification tasks: task name -> list of group names. The label of
#: a sample is its index into this list (cyclic group first, label 0).
TASKS: dict[str, list[str]] = {
    "quartic": ["C4", "C2^2"],
    "sextic": ["C6", "S3"],
    "octic_abelian": ["C8", "C4xC2", "C2^3"],
    "octic_nonabelian": ["D4", "Q8"],
    "nonic": ["C9", "C3^2"],
    "decic": ["C10", "D5"],
    # held-out variants (contamination-resistant; not in the paper)
    "degree14": ["C14", "D7"],
    "degree21": ["C21", "C7:C3"],
    "degree25": ["C25", "C5^2"],
}

#: Tasks analyzed in arXiv:2508.06670 (their criteria are published and
#: may be in LLM training data). The complement is held out.
PAPER_TASKS: frozenset[str] = frozenset(
    ["quartic", "sextic", "octic_abelian", "octic_nonabelian", "nonic", "decic"]
)
