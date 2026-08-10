import numpy as np
import pytest

from galois_ml.finite_field import GF


@pytest.fixture(scope="module", params=[(2, 4), (3, 3), (5, 4)])
def field(request):
    p, n = request.param
    return GF(p, n, seed=0)


def test_field_axioms_on_samples(field):
    rng = np.random.default_rng(0)
    for _ in range(20):
        a, b, c = (field.random_element(rng) for _ in range(3))
        # commutativity and associativity of mul
        assert np.array_equal(field.mul(a, b), field.mul(b, a))
        assert np.array_equal(
            field.mul(field.mul(a, b), c), field.mul(a, field.mul(b, c))
        )
        # distributivity
        assert np.array_equal(
            field.mul(a, field.add(b, c)),
            field.add(field.mul(a, b), field.mul(a, c)),
        )
        # inverses
        if a.any():
            assert np.array_equal(field.mul(a, field.inverse(a)), field.one())


def test_frobenius_is_field_automorphism(field):
    rng = np.random.default_rng(1)
    for _ in range(10):
        a, b = field.random_element(rng), field.random_element(rng)
        fa, fb = field.frobenius(a), field.frobenius(b)
        assert np.array_equal(field.frobenius(field.add(a, b)), field.add(fa, fb))
        assert np.array_equal(field.frobenius(field.mul(a, b)), field.mul(fa, fb))


def test_frobenius_has_order_n(field):
    rng = np.random.default_rng(2)
    a = field.random_element(rng)
    x = a.copy()
    for _ in range(field.n):
        x = field.frobenius(x)
    assert np.array_equal(x, a)  # sigma^n = id


def test_every_element_satisfies_x_q_equals_x(field):
    rng = np.random.default_rng(3)
    for _ in range(10):
        a = field.random_element(rng)
        assert np.array_equal(field.pow(a, field.q), a)


def test_trace_and_norm_are_galois_invariant(field):
    rng = np.random.default_rng(4)
    for _ in range(10):
        a = field.random_element(rng)
        fa = field.frobenius(a)
        assert field.trace(a) == field.trace(fa)
        assert field.norm(a) == field.norm(fa)


def test_normal_basis_frobenius_is_cyclic_shift(field):
    nb = field.normal_basis(seed=0)
    rng = np.random.default_rng(5)
    for _ in range(10):
        a = field.random_element(rng)
        c = nb.to_normal(a)
        # round trip
        assert np.array_equal(nb.from_normal(c), a)
        # Frobenius in normal coordinates is roll by 1
        c_frob = nb.to_normal(field.frobenius(a))
        assert np.array_equal(c_frob, np.roll(c, 1))


def test_is_square_counts(field):
    if field.p == 2:
        return  # everything is a square in characteristic 2
    squares = sum(field.is_square(e) for e in field.all_elements())
    assert squares == (field.q - 1) // 2 + 1  # half the units, plus zero
