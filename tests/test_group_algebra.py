import numpy as np
import pytest

from galois_ml.group_algebra import REALIZATIONS, FiniteGroup
from galois_ml.groups import GROUPS


@pytest.mark.parametrize("name", sorted(REALIZATIONS))
def test_group_axioms(name):
    g = REALIZATIONS[name]
    t = g.table
    n = g.order
    # identity
    assert np.array_equal(t[0], np.arange(n))
    assert np.array_equal(t[:, 0], np.arange(n))
    # each row/column is a permutation (Latin square == cancellation laws)
    for i in range(n):
        assert sorted(t[i]) == list(range(n))
        assert sorted(t[:, i]) == list(range(n))
    # associativity
    for a in range(n):
        for b in range(n):
            for c in range(n):
                assert t[t[a, b], c] == t[a, t[b, c]]
    # inverses exist
    for i in range(n):
        j = g.inverse(i)
        assert t[i, j] == 0 and t[j, i] == 0


@pytest.mark.parametrize("name", sorted(REALIZATIONS))
def test_realizations_match_group_registry(name):
    """The Cayley tables must reproduce the element-order statistics
    recorded in galois_ml.groups (which drive Chebotarev sampling)."""
    concrete = REALIZATIONS[name]
    abstract = GROUPS[name]
    assert concrete.order == abstract.order
    assert concrete.element_orders() == abstract.element_orders


def test_q8_is_not_d4():
    q8 = REALIZATIONS["Q8"]
    d4 = REALIZATIONS["D4"]
    assert q8.element_orders() == {1: 1, 2: 1, 4: 6}
    assert d4.element_orders() == {1: 1, 2: 5, 4: 2}


def test_left_translation_is_group_action():
    g = REALIZATIONS["S3"]
    rng = np.random.default_rng(0)
    x = rng.normal(size=g.order)
    for a in range(g.order):
        for b in range(g.order):
            # L_a (L_b x) == L_{ab} x
            lb = x[g.left_translation(b)]
            lab = lb[g.left_translation(a)]
            direct = x[g.left_translation(int(g.table[a, b]))]
            assert np.allclose(lab, direct)
