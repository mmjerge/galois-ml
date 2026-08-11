"""Offline tests for the multi-agent testbed: data-analysis helpers and
tool construction (agent execution itself requires Bedrock)."""

import numpy as np
import pytest

from galois_ml.benchmark import load_benchmark
from galois_ml.multi_agent import (
    build_tools,
    coefficient_distribution,
    labeled_examples,
    prime_power_indices,
    validate_rule,
)


@pytest.fixture(scope="module")
def bench():
    return load_benchmark(
        "quartic", n_train=300, n_test=100, n_coeffs=200, bayes_samples=10000
    )


def test_coefficient_distribution(bench):
    out = coefficient_distribution(bench, 4)
    assert out.startswith("a_4 distribution")
    assert "C4" in out and "C2^2" in out
    # a_1 is constant 1 for both classes
    assert "1: 1.000" in coefficient_distribution(bench, 1)
    assert "error" in coefficient_distribution(bench, 9999)


def test_prime_power_indices(bench):
    out = prime_power_indices(bench)
    assert "4 = 2^2" in out and "27 = 3^3" in out and "199" in out
    assert "6" not in out.split(": ")[1].split(", ")[0]  # 6 is not listed first


def test_validate_rule_scores_good_and_bad(bench):
    # the theorem rule: C4 iff some a_{p^2} = 0
    good = (
        "def predict(zeta):\n"
        "    def is_prime(m):\n"
        "        return m > 1 and all(m % d for d in range(2, int(m**0.5) + 1))\n"
        "    p = 2\n"
        "    while p * p <= len(zeta):\n"
        "        if is_prime(p) and zeta[p * p - 1] == 0:\n"
        "            return 0\n"
        "        p += 1\n"
        "    return 1\n"
    )
    out = validate_rule(bench, good)
    acc = float(out.split("validation accuracy: ")[1].split(" ")[0])
    assert acc > 0.95
    # constant rule scores near the class balance and reports mistakes
    out_bad = validate_rule(bench, "def predict(zeta):\n    return 0")
    assert "misclassified" in out_bad
    # broken code is reported, not raised
    assert "failed to execute" in validate_rule(bench, "def predict(z): raise ValueError")


def test_labeled_examples(bench):
    out = labeled_examples(bench, 0)
    assert "C4" in out and "a_4=" in out
    assert "error" in labeled_examples(bench, 7)


def test_build_tools_returns_strands_tools(bench):
    tools = build_tools(bench)
    assert len(tools) == 4
    names = {getattr(t, "tool_name", getattr(t, "__name__", "")) for t in tools}
    assert "test_rule" in names and "inspect_coefficient" in names
