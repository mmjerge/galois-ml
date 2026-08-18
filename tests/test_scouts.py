"""Offline tests for the multi-source scout setup.

Verifies the properties the experiment depends on: overlapping coverage so
lies are falsifiable, a persistent liar identity, lie rates that are
actually honored, and falsification that is individually plausible.
"""

import numpy as np
import pytest

from galois_ml.benchmark import load_benchmark
from galois_ml.multi_agent import coefficient_distribution
from galois_ml.scouts import (
    SCOUT_NAMES,
    build_scout_tools,
    falsify,
    scout_coverage,
)


@pytest.fixture(scope="module")
def bench():
    return load_benchmark(
        "sextic", n_train=200, n_test=100, n_coeffs=200, bayes_samples=4000
    )


def _tool(tools, name):
    for t in tools:
        if getattr(t, "tool_name", getattr(t, "__name__", "")) == name:
            return t
    raise AssertionError(f"no tool {name}")


def test_coverage_overlaps_so_lies_are_falsifiable(bench):
    cov = scout_coverage(bench)
    counts = {}
    for scout, idxs in cov.items():
        for n in idxs:
            counts[n] = counts.get(n, 0) + 1
    # every covered index has at least two reporters, so any single lie
    # can in principle be contradicted by a peer
    assert counts and min(counts.values()) >= 2
    # but not all four, or the task degenerates into majority voting
    assert max(counts.values()) < len(SCOUT_NAMES)


def test_falsify_keeps_numbers_but_swaps_classes(bench):
    honest = coefficient_distribution(bench, 4)
    lie = falsify(honest, bench.class_names)
    assert lie != honest
    body_h = sorted(s.split(": ", 1)[1] for s in honest.split(" -- ")[1].split(" | "))
    body_l = sorted(s.split(": ", 1)[1] for s in lie.split(" -- ")[1].split(" | "))
    # same distributions, reattributed: individually plausible
    assert body_h == body_l


def test_only_the_designated_scout_lies(bench):
    stats = {}
    tools = build_scout_tools(bench, lie_rate=1.0, liar="scout_c", seed=0,
                              stats=stats)
    cov = scout_coverage(bench)
    n = cov["scout_c"][0]
    honest = coefficient_distribution(bench, n)
    # the liar always lies at rate 1.0
    assert honest not in str(_tool(tools, "scout_c")(n=n))
    # a peer covering the same index tells the truth
    peer = next(s for s in SCOUT_NAMES if s != "scout_c" and n in cov[s])
    assert honest in str(_tool(tools, peer)(n=n))


def test_lie_rate_is_honored(bench):
    for rate in (0.0, 0.25, 1.0):
        stats = {}
        tools = build_scout_tools(bench, lie_rate=rate, liar="scout_c",
                                  seed=3, stats=stats)
        n = scout_coverage(bench)["scout_c"][0]
        tool = _tool(tools, "scout_c")
        for _ in range(40):
            tool(n=n)
        observed = stats["lies"] / 40
        assert abs(observed - rate) < 0.25, (rate, observed)


def test_scout_declines_uncovered_index(bench):
    cov = scout_coverage(bench)
    tools = build_scout_tools(bench, lie_rate=0.0, stats={})
    uncovered = next(n for n in cov["scout_b"] if n not in cov["scout_a"])
    out = str(_tool(tools, "scout_a")(n=uncovered))
    assert "do not cover" in out
    # and it points at someone who does, so overlap is discoverable
    assert any(s in out for s in SCOUT_NAMES)


def test_coverage_lookup_tool(bench):
    tools = build_scout_tools(bench, lie_rate=0.0, stats={})
    n = scout_coverage(bench)["scout_a"][0]
    out = str(_tool(tools, "which_scouts_cover")(n=n))
    assert "scout_a" in out


def test_call_accounting(bench):
    stats = {}
    tools = build_scout_tools(bench, lie_rate=0.0, stats=stats)
    n = scout_coverage(bench)["scout_a"][0]
    _tool(tools, "scout_a")(n=n)
    _tool(tools, "scout_a")(n=n)
    assert stats["calls"]["scout_a"] == 2
    assert stats["calls"]["scout_b"] == 0
