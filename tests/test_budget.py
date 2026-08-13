"""Tests for tool-call budgets.

The behaviour that matters: on exhaustion the tools must instruct the
agent to finalize rather than raising, so a capped run still yields a
scoreable answer instead of being lost.
"""

import pytest

from galois_ml.benchmark import load_benchmark
from galois_ml.budget import DEFAULT_MAX_TOOL_CALLS, ToolBudget
from galois_ml.multi_agent import build_tools


def test_default_cap_is_100():
    assert DEFAULT_MAX_TOOL_CALLS == 100
    assert ToolBudget().limit == 100


def test_budget_allows_then_blocks():
    b = ToolBudget(limit=3)
    for i in range(3):
        assert b.charge("test_rule") is None, f"call {i} should be allowed"
    assert b.used == 3
    assert b.exhausted
    msg = b.charge("test_rule")
    assert msg is not None
    # the message must tell the agent to finalize, not merely refuse
    assert "final" in msg.lower()
    assert "3/3" in msg
    # refused calls do not increment usage, but are counted separately
    assert b.used == 3
    assert b.exhausted_hits == 1


def test_unlimited_budget():
    for limit in (None, 0, -1):
        b = ToolBudget(limit=limit)
        assert b.unlimited
        for _ in range(50):
            assert b.charge("x") is None
        assert not b.exhausted
        assert b.used == 50


def test_per_tool_accounting():
    b = ToolBudget(limit=10)
    b.charge("inspect_coefficient")
    b.charge("inspect_coefficient")
    b.charge("test_rule")
    s = b.summary()
    assert s["per_tool"] == {"inspect_coefficient": 2, "test_rule": 1}
    assert s["used"] == 3 and s["limit"] == 10 and s["exhausted"] is False


def test_tools_return_finalize_message_when_capped():
    """The wired tools must degrade to the instruction, not raise."""
    bench = load_benchmark(
        "quartic", n_train=60, n_test=20, n_coeffs=100, bayes_samples=2000
    )
    budget = ToolBudget(limit=2)
    tools = build_tools(bench, budget=budget)
    by_name = {
        getattr(t, "tool_name", getattr(t, "__name__", "")): t for t in tools
    }
    inspect = by_name["inspect_coefficient"]

    first = str(inspect(n=4))
    assert "distribution" in first  # real data while budget remains
    second = str(inspect(n=9))
    assert "distribution" in second
    # budget now spent: every tool should return the finalize instruction
    for name, tool in by_name.items():
        kwargs = {"inspect_coefficient": {"n": 4}, "show_examples": {"label": 0},
                  "test_rule": {"code": "def predict(zeta):\n    return 0"},
                  "list_prime_powers": {}}[name]
        out = str(tool(**kwargs) if kwargs else tool())
        assert "BUDGET EXHAUSTED" in out, f"{name} did not honor the budget"
        assert "final" in out.lower()


def test_capped_run_is_recorded_in_summary():
    b = ToolBudget(limit=1)
    b.charge("a")
    b.charge("a")  # refused
    s = b.summary()
    assert s["exhausted"] is True
    assert s["refused_calls"] == 1
