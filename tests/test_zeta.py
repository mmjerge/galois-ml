import numpy as np
import pytest

from galois_ml.zeta import euler_coefficient, primes_up_to, zeta_coefficients


def test_euler_coefficient_split_prime():
    # p split completely (f=1): a_p = g = degree.
    assert euler_coefficient(1, 1, 4) == 4
    # a_{p^2} = C(g+1, g-1); for g=4: C(5,3) = 10.
    assert euler_coefficient(2, 1, 4) == 10


def test_euler_coefficient_inert_prime():
    # p inert in a quartic field (f=4, g=1): a_p = a_{p^2} = a_{p^3} = 0.
    assert euler_coefficient(1, 4, 1) == 0
    assert euler_coefficient(2, 4, 1) == 0
    assert euler_coefficient(3, 4, 1) == 0
    assert euler_coefficient(4, 4, 1) == 1


def test_euler_coefficient_matches_paper_table2():
    # Nonic (degree 9) values from Table 2 of the paper: a_{p^3}.
    assert euler_coefficient(3, 1, 9) == 165  # split, (1,1,9)
    assert euler_coefficient(3, 3, 3) == 3    # (1,3,3)
    assert euler_coefficient(3, 9, 1) == 0    # inert, (1,9,1)


def test_zeta_coefficients_gaussian_integers():
    # K = Q(i): f(p) = 1 if p = 1 (mod 4) else 2 (for odd p).
    # a_n counts ideals of norm n in Z[i]; check odd n by hand.
    N = 100
    frob = {p: (1 if p % 4 == 1 else 2) for p in primes_up_to(N)}
    a = zeta_coefficients(N, frob, degree=2)

    def a_n(n):
        return a[n - 1]

    assert a_n(1) == 1
    assert a_n(3) == 0    # 3 inert
    assert a_n(9) == 1    # (3)^2 is the only ideal of norm 9
    assert a_n(5) == 2    # 5 splits
    assert a_n(25) == 3
    assert a_n(65) == 4   # 5 and 13 both split: 2 * 2
    assert a_n(45) == 2   # 45 = 9 * 5: 1 * 2


def test_zeta_multiplicativity():
    N = 1000
    rng = np.random.default_rng(1)
    primes = primes_up_to(N)
    frob = {p: int(rng.choice([1, 2, 3, 6])) for p in primes}
    a = zeta_coefficients(N, frob, degree=6)

    def a_n(n):
        return a[n - 1]

    # coprime multiplicativity on a few pairs
    for m, n in [(4, 9), (8, 27), (25, 27), (11, 13)]:
        assert a_n(m * n) == a_n(m) * a_n(n)


def test_invalid_inertia_degree_raises():
    N = 10
    frob = {p: 5 for p in primes_up_to(N)}  # 5 does not divide 4
    with pytest.raises(ValueError):
        zeta_coefficients(N, frob, degree=4)
