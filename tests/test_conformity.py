"""Tests for the correlated-failure analysis.

The statistics are validated against synthetic agents with known
correlation structure, so the metrics are checked against cases where
the right answer is derivable by hand.
"""

import numpy as np
import pytest

from galois_ml.benchmark import load_benchmark
from galois_ml.conformity import (
    AgentRun,
    analyze_runs,
    effective_independent_count,
    independent_ensemble_accuracy,
    majority_vote_accuracy,
    pairwise_error_overlap,
)


def test_identical_agents_are_maximally_correlated():
    # three agents failing on exactly the same items
    mask = np.zeros(100, dtype=bool)
    mask[:20] = True
    out = pairwise_error_overlap([mask.copy() for _ in range(3)])
    assert out["observed_jaccard"] == pytest.approx(1.0)
    # independence would predict a far smaller overlap at a 20% error rate
    assert out["independent_jaccard"] < 0.15
    assert out["correlation_ratio"] > 5
    assert out["n_pairs"] == 3


def test_disjoint_agents_are_anticorrelated():
    a = np.zeros(100, dtype=bool); a[:20] = True
    b = np.zeros(100, dtype=bool); b[20:40] = True
    out = pairwise_error_overlap([a, b])
    assert out["observed_jaccard"] == pytest.approx(0.0)
    assert out["correlation_ratio"] == pytest.approx(0.0)


def test_random_independent_agents_have_ratio_near_one():
    rng = np.random.default_rng(0)
    masks = [rng.random(4000) < 0.25 for _ in range(4)]
    out = pairwise_error_overlap(masks)
    # genuinely independent failures: observed ~ expected
    assert 0.85 < out["correlation_ratio"] < 1.15


def test_majority_vote_beats_individuals_when_errors_differ():
    y = np.zeros(9, dtype=np.int64)
    # each agent wrong on a different third; majority is always right
    p1 = y.copy(); p1[0:3] = 1
    p2 = y.copy(); p2[3:6] = 1
    p3 = y.copy(); p3[6:9] = 1
    assert majority_vote_accuracy([p1, p2, p3], y) == pytest.approx(1.0)
    # identical agents: the vote inherits the shared error exactly
    assert majority_vote_accuracy([p1, p1, p1], y) == pytest.approx(6 / 9)


def test_independent_ensemble_accuracy_matches_theory():
    # 3 independent agents at 30% error: majority wrong iff >=2 err
    # P = 3*(0.3^2)(0.7) + 0.3^3 = 0.189 + 0.027 = 0.216 -> acc 0.784
    acc = independent_ensemble_accuracy([0.3, 0.3, 0.3], n_items=400_000, seed=1)
    assert acc == pytest.approx(0.784, abs=0.01)


def test_effective_independent_count_detects_illusory_redundancy():
    # observed majority accuracy equal to a single agent's accuracy
    # means the ensemble was worth ~1 independent agent
    rates = [0.2] * 5
    single_acc = 0.8
    eff = effective_independent_count(single_acc, rates)
    assert eff == pytest.approx(1.0, abs=0.3)
    # a genuinely better ensemble should score above 1
    better = independent_ensemble_accuracy([0.2] * 5, n_items=200_000, seed=2)
    assert effective_independent_count(better, rates) > 3


def test_analyze_runs_end_to_end():
    bench = load_benchmark(
        "quartic", n_train=60, n_test=40, n_coeffs=100, bayes_samples=4000
    )
    n = len(bench.y_test)
    # two agents with identical errors, one with different errors
    shared = np.zeros(n, dtype=bool); shared[:6] = True
    other = np.zeros(n, dtype=bool); other[6:12] = True
    runs, preds = [], []
    for i, mask in enumerate([shared, shared.copy(), other]):
        p = bench.y_test.copy()
        p[mask] = 1 - p[mask]  # flip where wrong (binary task)
        runs.append(AgentRun(
            seed=i, valid=True, accuracy=float(1 - mask.mean()),
            error_mask=mask.tolist(), conjecture=f"rule {i % 2}",
        ))
        preds.append(p)
    m = analyze_runs(runs, bench, preds)
    assert m["n_valid_agents"] == 3
    assert m["correlation_ratio"] > 1  # two of three fail identically
    assert m["distinct_conjectures"] == 2
    assert "effective_independent_agents" in m
    assert m["majority_vote_accuracy"] >= 0.0


def test_analyze_runs_handles_too_few_valid():
    bench = load_benchmark(
        "quartic", n_train=40, n_test=20, n_coeffs=100, bayes_samples=2000
    )
    runs = [AgentRun(seed=0, valid=False, error="boom")]
    m = analyze_runs(runs, bench, [np.full(20, -1)])
    assert "note" in m
