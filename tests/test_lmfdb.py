"""Offline tests for the polynomial -> zeta coefficient path.

The API-fetching part is exercised separately (network); here we verify
the mathematical core against cyclotomic fields, whose decomposition law
is classical: in Q(zeta_m), an unramified prime p has inertia degree
equal to the multiplicative order of p mod m.
"""

import numpy as np

from galois_ml.lmfdb import decomposition_from_polynomial
from galois_ml.zeta import primes_up_to, zeta_coefficients


def _order_mod(p: int, m: int) -> int:
    k, x = 1, p % m
    while x != 1:
        x = (x * p) % m
        k += 1
    return k


def test_cyclotomic_q_zeta5():
    # Q(zeta_5): x^4 + x^3 + x^2 + x + 1, Galois group C4.
    coeffs = [1, 1, 1, 1, 1]  # ascending
    N = 200
    dec = decomposition_from_polynomial(coeffs, N)
    for p in primes_up_to(N):
        f, g = dec[p]
        if p == 5:
            # totally ramified: (5) = (1 - zeta_5)^4
            assert (f, g) == (1, 1)
        else:
            expected_f = _order_mod(p, 5)
            assert f == expected_f, f"p={p}: f={f}, expected {expected_f}"
            assert g == 4 // expected_f


def test_cyclotomic_q_zeta8():
    # Q(zeta_8): x^4 + 1, Galois group C2 x C2; p=2 totally ramified.
    coeffs = [1, 0, 0, 0, 1]
    N = 200
    dec = decomposition_from_polynomial(coeffs, N)
    for p in primes_up_to(N):
        f, g = dec[p]
        if p == 2:
            assert (f, g) == (1, 1)
        else:
            expected_f = _order_mod(p, 8)  # always 1 or 2
            assert f == expected_f
            assert g == 4 // expected_f


def test_zeta_coefficients_with_ramified_primes():
    # Q(zeta_5) again: a_5 = 1 (unique ideal of norm 5), a_25 = 1,
    # a_{5k} = a_5 * a_k for (k, 5) = 1.
    coeffs = [1, 1, 1, 1, 1]
    N = 200
    dec = decomposition_from_polynomial(coeffs, N)
    a = zeta_coefficients(N, dec, degree=4)

    def a_n(n):
        return a[n - 1]

    assert a_n(5) == 1
    assert a_n(25) == 1
    # p = 11 splits completely (11 = 1 mod 5): a_11 = 4
    assert a_n(11) == 4
    assert a_n(55) == 4  # multiplicativity across the ramified prime
    # p = 2 has order 4 mod 5: inert, a_2 = a_4 = a_8 = 0, a_16 = 1
    assert a_n(2) == 0 and a_n(4) == 0 and a_n(8) == 0 and a_n(16) == 1
