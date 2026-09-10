"""Follow-up tests requested by the external review.

Three verification gaps the review called load-bearing:

1. `ChebotarevBayesClassifier.log_likelihoods` had never been checked
   against independently enumerated small cases, and every `certify.py`
   quantity rests on it.
2. The attainable sets come from the same closed-form binomial formula
   everywhere; an independent derivation (expanding the Euler factor as
   an actual power series) should agree.
3. Soundness claims ("honest generator output is never flagged /
   rejected") had been spot-checked on two tasks, not swept across the
   suite.
"""

import numpy as np
import pytest

from galois_ml.benchmark import load_benchmark
from galois_ml.certify import admissible_mask, exact_posterior
from galois_ml.criteria import ChebotarevBayesClassifier
from galois_ml.groups import GROUPS, TASKS
from galois_ml.guard import ToolGuard, attainable_prime_power_values
from galois_ml.multi_agent import coefficient_distribution, validate_rule
from galois_ml.zeta import euler_coefficient


# -- 1. log_likelihoods against independent enumeration ---------------------

def _series_coefficient(f: int, g: int, k: int) -> int:
    """Coefficient of T^k in (1 - T^f)^(-g), by actual series expansion
    (repeated convolution), independent of the binomial closed form."""
    # (1 - T^f)^(-1) = 1 + T^f + T^2f + ...
    base = [1 if i % f == 0 else 0 for i in range(k + 1)]
    series = [1] + [0] * k
    for _ in range(g):
        series = [
            sum(series[j] * base[i - j] for j in range(i + 1))
            for i in range(k + 1)
        ]
    return series[k]


def test_euler_coefficient_matches_series_expansion():
    for f in (1, 2, 3, 4, 6, 7):
        for g in (1, 2, 3, 6):
            for k in range(0, 9):
                assert euler_coefficient(k, f, g) == _series_coefficient(
                    f, g, k
                ), (f, g, k)


def test_attainable_sets_match_series_expansion():
    for degree in (4, 6, 9, 10, 21):
        for k in range(1, 7):
            expected = {
                _series_coefficient(f, degree // f, k)
                for f in range(1, degree + 1) if degree % f == 0
            }
            assert attainable_prime_power_values(degree, k) == expected


def _enumerated_log_likelihoods(task, n_coeffs, row):
    """Independent reimplementation: for each prime, push the group's
    Frobenius-order measure through the signature map by direct
    enumeration, look up the row's signature, and sum logs. Written
    from the model definition, not from the classifier's code."""
    import math

    from galois_ml.zeta import primes_up_to

    groups = [GROUPS[g] for g in TASKS[task]]
    degree = groups[0].order
    out = []
    for group in groups:
        orders, probs = group.frobenius_orders
        total = 0.0
        for p in primes_up_to(n_coeffs):
            K = max(k for k in range(1, degree + 1) if p**k <= n_coeffs)
            sig = tuple(int(row[p**k - 1]) for k in range(1, K + 1))
            density = sum(
                float(prob) for f, prob in zip(orders, probs)
                if tuple(
                    euler_coefficient(k, int(f), group.order // int(f))
                    for k in range(1, K + 1)
                ) == sig
            )
            # mirror the classifier's documented skip: a signature no
            # candidate can produce is treated as ramified and ignored
            all_zero = all(
                sum(
                    float(pr) for f2, pr in zip(*g2.frobenius_orders)
                    if tuple(
                        euler_coefficient(k, int(f2), g2.order // int(f2))
                        for k in range(1, K + 1)
                    ) == sig
                ) == 0.0
                for g2 in groups
            )
            if all_zero:
                continue
            total += math.log(density) if density > 0 else -math.inf
        out.append(total)
    return np.array(out)


@pytest.mark.parametrize("task", ["quartic", "sextic", "degree21"])
def test_log_likelihoods_match_independent_enumeration(task):
    n_coeffs = 60
    bench = load_benchmark(
        task, n_train=40, n_test=20, n_coeffs=n_coeffs, bayes_samples=2000
    )
    clf = ChebotarevBayesClassifier(task, n_coeffs)
    for row in bench.X_test[:10]:
        got = clf.log_likelihoods(row)
        want = _enumerated_log_likelihoods(task, n_coeffs, row)
        for a, b in zip(got, want):
            if np.isinf(a) or np.isinf(b):
                assert a == b, (task, a, b)
            else:
                assert a == pytest.approx(b, abs=1e-9), (task, a, b)


def test_log_likelihoods_known_inert_prime_case():
    """Hand-computed: an all-zero signature at p=2 (a_2=a_4=a_8=0) is an
    inert prime, possible for C4 (f=4) and impossible for C2^2 (whose
    residue degrees are 1 and 2). The C2^2 likelihood must be -inf."""
    n_coeffs = 10
    clf = ChebotarevBayesClassifier("quartic", n_coeffs)
    row = np.zeros(n_coeffs)
    row[0] = 1.0
    # p=2 inert: a_2=a_4=a_8=0 (already zero). Give p=3,5,7 signatures
    # possible for both classes: f=2 -> a_p=0, a_{p^2}=2.
    row[8] = 2.0   # a_9, so p=3 has f=2
    row[4] = 0.0   # a_5: f in {2,4} for both... f=2 gives a_5=0 (K=1)
    row[6] = 0.0   # a_7 likewise
    lls = clf.log_likelihoods(row)
    assert np.isfinite(lls[0])       # C4 explains everything
    assert np.isneginf(lls[1])       # C2^2 cannot produce an inert prime
    post = exact_posterior("quartic", row.reshape(1, -1), n_coeffs=n_coeffs)
    assert post[0, 0] == pytest.approx(1.0)


# -- 3. soundness sweeps across the full task suite --------------------------

@pytest.mark.parametrize("task", sorted(TASKS))
def test_honest_data_always_admissible(task):
    bench = load_benchmark(
        task, n_train=60, n_test=40, n_coeffs=120, bayes_samples=2000
    )
    degree = GROUPS[bench.class_names[0]].order
    assert admissible_mask(bench.X_train, degree).all()
    assert admissible_mask(bench.X_test, degree).all()


@pytest.mark.parametrize("task", sorted(TASKS))
def test_guard_soundness_sweep(task):
    """Honest tool outputs across every task and a spread of indices,
    including composite and high prime-power indices, are never flagged."""
    bench = load_benchmark(
        task, n_train=60, n_test=40, n_coeffs=120, bayes_samples=2000
    )
    guard = ToolGuard(
        degree=GROUPS[bench.class_names[0]].order,
        n_coeffs=120,
        class_names=bench.class_names,
        recompute_validation=lambda code: validate_rule(bench, code),
    )
    for n in (1, 2, 8, 12, 27, 30, 64, 100, 120):
        honest = coefficient_distribution(bench, n)
        assert guard.filter_distribution(honest, n) == honest, (task, n)
    assert not guard.detected
