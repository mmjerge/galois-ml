"""Dedekind zeta coefficient machinery.

Implements Proposition 2.1 / Corollary 2.2 of Lee & Lee (arXiv:2508.06670):
for a Galois extension K/Q in which the rational prime p has decomposition
type (e, f, g), the p-th Euler factor of the Dedekind zeta function is

    zeta_{K,p}(s) = (1 - p^{-f s})^{-g},

so the p^a-th zeta coefficient is

    a_{p^a}(K) = C(g + a/f - 1, g - 1)   if f | a,   else 0.

Zeta coefficients are multiplicative (a_{mn} = a_m a_n for coprime m, n),
which lets us assemble a_n for all n <= N from decomposition data at each
prime.

We model the *unramified* part only (e = 1): by Chebotarev, f is the order
of the Frobenius class at p, and g = [K:Q] / f. Ramified primes are a
density-zero set and are treated as unramified here; the theorem-based
criteria in `galois_ml.criteria` remain valid because they rely on
decomposition types that already occur among unramified primes.
"""

from __future__ import annotations

from functools import lru_cache
from math import comb, isqrt

import numpy as np


@lru_cache(maxsize=None)
def _factor_table(N: int):
    """Sieve-based factorization helper for 1..N.

    Returns (primes, decomp) where primes is the list of primes <= N and
    decomp[n] = (m, p, a) with n = m * p^a, p = smallest prime factor of n
    and gcd(m, p) = 1. decomp[0] = decomp[1] = None.
    """
    spf = list(range(N + 1))
    for i in range(2, isqrt(N) + 1):
        if spf[i] == i:  # i is prime
            for j in range(i * i, N + 1, i):
                if spf[j] == j:
                    spf[j] = i
    primes = [i for i in range(2, N + 1) if spf[i] == i]
    decomp: list = [None] * (N + 1)
    for n in range(2, N + 1):
        p = spf[n]
        m, a = n, 0
        while m % p == 0:
            m //= p
            a += 1
        decomp[n] = (m, p, a)
    return primes, decomp


def primes_up_to(N: int) -> list[int]:
    """All rational primes <= N."""
    return list(_factor_table(N)[0])


def euler_coefficient(a: int, f: int, g: int) -> int:
    """Coefficient of p^{-as} in the Euler factor (1 - p^{-fs})^{-g}.

    This is a_{p^a}(K) for a prime with inertia degree f and g primes
    above p (Corollary 2.2 of the paper).
    """
    if a < 0:
        raise ValueError("power a must be nonnegative")
    if a % f != 0:
        return 0
    k = a // f
    return comb(g + k - 1, g - 1)


def zeta_coefficients(
    N: int, decomposition: dict[int, int | tuple[int, int]], degree: int
) -> np.ndarray:
    """Zeta coefficients a_1(K), ..., a_N(K) of a degree-`degree` Galois
    extension K/Q, given decomposition data for each prime p <= N:

    * an int f - the inertia degree, assuming p unramified
      (e = 1, g = degree / f), or
    * a pair (f, g) - inertia degree and number of primes above p,
      which also covers ramified primes (e = degree / (f g) does not
      enter the Euler factor; see Prop 2.1 of the paper).

    Returns a float array of length N; index i holds a_{i+1}(K).
    """
    _, decomp = _factor_table(N)
    coeffs = [0.0, 1.0] + [0.0] * (N - 1)  # coeffs[n] = a_n, a_1 = 1
    pk_cache: dict[tuple[int, int], int] = {}
    for n in range(2, N + 1):
        m, p, a = decomp[n]
        key = (p, a)
        c = pk_cache.get(key)
        if c is None:
            data = decomposition[p]
            if isinstance(data, tuple):
                f, g = data
            else:
                f, g = data, None
            if degree % f != 0:
                raise ValueError(
                    f"inertia degree f={f} at p={p} does not divide "
                    f"the extension degree {degree}"
                )
            if g is None:
                g = degree // f
            elif f * g > degree or degree % (f * g) != 0:
                raise ValueError(
                    f"invalid decomposition (f={f}, g={g}) at p={p} "
                    f"for degree {degree}"
                )
            c = euler_coefficient(a, f, g)
            pk_cache[key] = c
        coeffs[n] = coeffs[m] * c
    return np.array(coeffs[1:], dtype=np.float64)
