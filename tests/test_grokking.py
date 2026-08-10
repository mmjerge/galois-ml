import numpy as np
import torch

from galois_ml.grokking import (
    GFMultiplicationTask,
    GrokkingMLP,
    fourier_concentration,
    frobenius_alignment_zscore,
)


def _task():
    return GFMultiplicationTask(p=5, n=2)  # GF(25), unit group C_24


def test_discrete_log_table():
    task = _task()
    g = task.elements[task.generator_id]
    for i, e in enumerate(task.elements):
        k = int(task.dlog[i])
        assert np.array_equal(task.field.pow(g, k), e)
    # dlog is a bijection onto 0..q-2
    assert sorted(task.dlog) == list(range(task.group_order))


def test_product_labels_match_dlog_addition():
    task = _task()
    m = task.group_order
    for i in (0, 3, 7):
        for j in (1, 5, 11):
            k = task.product_id(i, j)
            assert (task.dlog[i] + task.dlog[j]) % m == task.dlog[k]


def test_frobenius_permutation_multiplies_dlog_by_p():
    task = _task()
    m = task.group_order
    for i in range(m):
        assert task.dlog[task.frobenius_perm[i]] == (task.p * task.dlog[i]) % m


def test_split_and_model_shapes():
    task = _task()
    X_tr, y_tr, X_te, y_te = task.split(train_fraction=0.5, seed=0)
    assert len(y_tr) + len(y_te) == task.group_order**2
    model = GrokkingMLP(task.group_order, d_embed=16, d_hidden=32)
    logits = model(X_tr[:7])
    assert logits.shape == (7, task.group_order)


def test_fourier_concentration_detects_structure():
    task = _task()
    m = task.group_order
    dlog = task.dlog
    rng = np.random.default_rng(0)
    # planted single-frequency embedding: near-total concentration
    k = dlog.astype(float)
    planted = np.stack(
        [np.cos(2 * np.pi * 3 * k / m), np.sin(2 * np.pi * 3 * k / m)], axis=1
    )
    assert fourier_concentration(planted, dlog) > 0.99
    # random embedding: flat spectrum, low concentration
    random = rng.normal(size=(m, 32))
    assert fourier_concentration(random, dlog) < 0.7


def test_frobenius_alignment_on_planted_embedding():
    task = _task()
    m = task.group_order
    k = task.dlog.astype(float)
    # Multi-frequency Fourier embedding (what a grokked net looks like).
    # The frequency set must be closed under multiplication by p mod m
    # (Frobenius sends frequency f to p*f): orbits {1,5} and {3,15}
    # under x5 mod 24. A non-closed set would make the Frobenius action
    # leave the column space, and no orthogonal map could realize it.
    cols = []
    for freq in (1, 5, 3, 15):
        cols += [np.cos(2 * np.pi * freq * k / m),
                 np.sin(2 * np.pi * freq * k / m)]
    E = np.stack(cols, axis=1)
    resid, baseline, z = frobenius_alignment_zscore(
        E, task.frobenius_perm, n_baseline=30
    )
    assert resid < 1e-6          # Frobenius is exactly orthogonal here
    assert baseline > 0.1        # random permutations are not
    assert z < -3
