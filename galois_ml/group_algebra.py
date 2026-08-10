"""Concrete finite groups via multiplication tables.

`galois_ml.groups` only records element-order statistics (enough for
Chebotarev sampling). For *equivariant architectures* we need the actual
group law: a G-equivariant linear map on the regular representation
R^{|G|} is precisely group convolution, W[h, k] = w(h^{-1} k), the
noncyclic generalization of a circulant matrix.

Groups are encoded by an |G| x |G| multiplication table over element
indices 0..|G|-1 with identity 0. Constructors cover every Galois group
in the paper: cyclic, dihedral, quaternion, and direct products.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FiniteGroup:
    name: str
    table: np.ndarray  # table[i, j] = index of g_i * g_j; identity is 0

    @property
    def order(self) -> int:
        return self.table.shape[0]

    def inverse(self, i: int) -> int:
        return int(np.where(self.table[i] == 0)[0][0])

    def element_order(self, i: int) -> int:
        k, x = 1, i
        while x != 0:
            x = int(self.table[x, i])
            k += 1
        return k

    def element_orders(self) -> dict[int, int]:
        """Element order -> count, for cross-checking `galois_ml.groups`."""
        counts: dict[int, int] = {}
        for i in range(self.order):
            o = self.element_order(i)
            counts[o] = counts.get(o, 0) + 1
        return counts

    def left_translation(self, g: int) -> np.ndarray:
        """Permutation L_g of the regular representation:
        (L_g x)[h] = x[g^{-1} h]. Returns the index array `perm` with
        (L_g x) = x[perm]."""
        g_inv = self.inverse(g)
        return self.table[g_inv]  # perm[h] = g^{-1} h

    def convolution_index(self) -> np.ndarray:
        """Index matrix idx[h, k] = h^{-1} k, so that a G-equivariant
        linear map has matrix W[h, k] = w[idx[h, k]]."""
        inv = np.array([self.inverse(i) for i in range(self.order)])
        return self.table[inv]  # row h is (h^{-1} k)_k


def cyclic(n: int) -> FiniteGroup:
    """C_n with elements 0..n-1 under addition mod n."""
    idx = np.arange(n)
    return FiniteGroup(f"C{n}", (idx[:, None] + idx[None, :]) % n)


def _pair_index(n: int, i: int, j: int) -> int:
    return j * n + i  # element a^i b^j  ->  index


def _binary_extension(name: str, n: int, twist: int) -> FiniteGroup:
    """Group <a, b | a^n = 1, b^2 = a^twist, b a b^-1 = a^-1>.

    twist = 0 gives the dihedral group D_n; n = 4, twist = 2 gives Q8.
    Elements are a^i b^j with 0 <= i < n, j in {0, 1}:
    (a^i1 b^j1)(a^i2 b^j2) = a^(i1 + (-1)^j1 i2 + twist*(j1 & j2)) b^(j1+j2 mod 2)
    """
    table = np.zeros((2 * n, 2 * n), dtype=np.int64)
    for j1 in range(2):
        for i1 in range(n):
            for j2 in range(2):
                for i2 in range(n):
                    sign = -1 if j1 else 1
                    i = (i1 + sign * i2 + twist * (j1 & j2)) % n
                    j = (j1 + j2) % 2
                    table[_pair_index(n, i1, j1), _pair_index(n, i2, j2)] = (
                        _pair_index(n, i, j)
                    )
    return FiniteGroup(name, table)


def dihedral(n: int) -> FiniteGroup:
    """D_n of order 2n (D_3 = S3)."""
    return _binary_extension(f"D{n}", n, twist=0)


def quaternion() -> FiniteGroup:
    """The quaternion group Q8."""
    return _binary_extension("Q8", 4, twist=2)


def direct_product(a: FiniteGroup, b: FiniteGroup) -> FiniteGroup:
    """A x B with index (i, j) -> j * |A| + i; identity stays at 0."""
    na, nb = a.order, b.order
    table = np.zeros((na * nb, na * nb), dtype=np.int64)
    for j1 in range(nb):
        for i1 in range(na):
            for j2 in range(nb):
                for i2 in range(na):
                    i, j = a.table[i1, i2], b.table[j1, j2]
                    table[j1 * na + i1, j2 * na + i2] = j * na + i
    return FiniteGroup(f"{a.name}x{b.name}", table)


#: Concrete realizations of every Galois group in `galois_ml.groups.GROUPS`.
REALIZATIONS: dict[str, FiniteGroup] = {
    "C4": cyclic(4),
    "C2^2": direct_product(cyclic(2), cyclic(2)),
    "C6": cyclic(6),
    "S3": dihedral(3),
    "C8": cyclic(8),
    "C4xC2": direct_product(cyclic(4), cyclic(2)),
    "C2^3": direct_product(direct_product(cyclic(2), cyclic(2)), cyclic(2)),
    "D4": dihedral(4),
    "Q8": quaternion(),
    "C9": cyclic(9),
    "C3^2": direct_product(cyclic(3), cyclic(3)),
    "C10": cyclic(10),
    "D5": dihedral(5),
}
