"""Tests for the final-artifact release gate (galois_ml.gate).

The claims under test map to the proposition's moving parts: the margin
formula, the release inequality, fail-closed behavior, the
candidate-count correction, and the end-to-end property that an
arbitrarily corrupted (here: label-inverted) candidate cannot be
released when it is worse than the fallback by more than epsilon.
"""

import numpy as np
import pytest

from galois_ml.gate import (
    GateDecision,
    certify_release,
    gate_rule_code,
    hoeffding_margin,
    theorem_fallback,
)

GOOD_RULE = (
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
# the pilot's signature failure: correct criterion, wrong class
INVERTED_RULE = GOOD_RULE.replace("return 0", "return TMP").replace(
    "return 1", "return 0"
).replace("return TMP", "return 1")
CRASHING_RULE = "def predict(zeta):\n    raise RuntimeError('boom')\n"


def test_margin_formula():
    assert hoeffding_margin(200, 0.05) == pytest.approx(
        np.sqrt(2 * np.log(20) / 200)
    )
    # K candidates: log(K/delta), computed stably
    assert hoeffding_margin(200, 0.05, n_candidates=4) == pytest.approx(
        np.sqrt(2 * np.log(4 / 0.05) / 200)
    )
    # monotone: more data or more risk tolerance -> smaller margin
    assert hoeffding_margin(800, 0.05) < hoeffding_margin(200, 0.05)
    assert hoeffding_margin(200, 0.2) < hoeffding_margin(200, 0.05)
    with pytest.raises(ValueError):
        hoeffding_margin(0, 0.05)
    with pytest.raises(ValueError):
        hoeffding_margin(100, 1.5)


def test_release_inequality_is_what_decides():
    y = np.zeros(1000, dtype=int)
    fb = np.zeros(1000, dtype=int)          # fallback perfect
    cand = np.zeros(1000, dtype=int)
    cand[:10] = 1                            # candidate 1% worse
    d = certify_release(cand, fb, y, epsilon=0.02, delta=0.05)
    # decision must equal the stated inequality, whatever it is
    accept = d.paired_loss_difference + d.margin <= d.epsilon
    assert (d.released == "candidate") == accept
    assert d.paired_loss_difference == pytest.approx(0.01)


def test_clearly_better_candidate_released():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 5000)
    cand = y.copy()                          # perfect candidate
    fb = y.copy()
    flip = rng.random(5000) < 0.2            # fallback 20% wrong
    fb[flip] = 1 - fb[flip]
    d = certify_release(cand, fb, y, epsilon=0.02, delta=0.05)
    assert d.released == "candidate"
    assert d.paired_loss_difference < 0


def test_worse_candidate_falls_back():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 5000)
    fb = y.copy()
    cand = y.copy()
    flip = rng.random(5000) < 0.2            # candidate 20% wrong
    cand[flip] = 1 - cand[flip]
    d = certify_release(cand, fb, y, epsilon=0.02, delta=0.05)
    assert d.released == "fallback"


def test_failed_candidate_fails_closed():
    y = np.zeros(100, dtype=int)
    d = certify_release(None, np.zeros(100, dtype=int), y)
    assert d.released == "fallback"
    assert d.candidate_holdout_accuracy is None


def test_candidate_count_tightens_the_bound():
    y = np.zeros(500, dtype=int)
    fb = np.zeros(500, dtype=int)
    cand = np.zeros(500, dtype=int)
    d1 = certify_release(cand, fb, y, n_candidates=1)
    d5 = certify_release(cand, fb, y, n_candidates=5)
    # family budget unchanged; per-candidate budget divided
    assert d5.delta_family == d1.delta_family
    assert d5.delta_per_candidate == pytest.approx(d1.delta_family / 5)
    assert d5.margin > d1.margin
    # the certificate speaks at family level, per review sec 3.2
    assert "family-level" in d5.certificate
    with pytest.raises(ValueError):
        certify_release(cand, fb, y, n_candidates=0)


def test_theorem_bug_inputs_rejected():
    """Review sec 3.4: negative epsilon voids the proposition (even the
    fallback violates the bound), and delta must be validated as given,
    not only after division."""
    y = np.zeros(100, dtype=int)
    p = np.zeros(100, dtype=int)
    with pytest.raises(ValueError):
        certify_release(p, p, y, epsilon=-0.01)
    with pytest.raises(ValueError):
        certify_release(p, p, y, epsilon=float("nan"))
    with pytest.raises(ValueError):
        certify_release(p, p, y, epsilon=1.5)
    with pytest.raises(ValueError):
        certify_release(p, p, y, delta=0.0)
    with pytest.raises(ValueError):
        certify_release(p, p, y[:0])  # empty labels
    with pytest.raises(ValueError):
        certify_release(p[:50], p, y)  # shape mismatch


def test_entropy_seed_and_fallback_provenance():
    from galois_ml.gate import gate_rule_code, theorem_fallback

    fb = theorem_fallback("quartic", 200)
    assert fb.fallback_kind == "theorem_criteria"
    d = gate_rule_code(
        "quartic", GOOD_RULE, fb, m=300, n_coeffs=200,
        holdout_seed=None,  # OS entropy
    )
    assert d.fallback_kind == "theorem_criteria"
    assert "holdout_seed=" in d.reason


def test_gate_end_to_end_good_rule_released():
    # epsilon must exceed the margin for a merely-equal candidate to
    # pass: certification at epsilon needs m >= 2 ln(1/delta)/epsilon^2
    # (~15k samples for epsilon=0.02), which the gate makes visible
    # rather than hiding. At m=800 the margin is ~0.087.
    d = gate_rule_code(
        "quartic", GOOD_RULE, theorem_fallback("quartic", 200),
        m=800, n_coeffs=200, epsilon=0.10, delta=0.05,
    )
    assert d.released == "candidate"
    assert d.candidate_holdout_accuracy > 0.95


def test_gate_sample_size_requirement_is_honest():
    """A candidate exactly matching the fallback is released iff the
    margin fits inside epsilon -- the gate must not release on
    insufficient data."""
    y = np.zeros(800, dtype=int)
    same = np.zeros(800, dtype=int)
    tight = certify_release(same, same, y, epsilon=0.02, delta=0.05)
    assert tight.released == "fallback"          # margin 0.087 > 0.02
    enough = certify_release(
        np.zeros(20_000, dtype=int), np.zeros(20_000, dtype=int),
        np.zeros(20_000, dtype=int), epsilon=0.02, delta=0.05,
    )
    assert enough.released == "candidate"        # margin ~0.017 <= 0.02


def test_gate_end_to_end_inverted_rule_blocked():
    """The pilot's confidently-wrong-but-structurally-sound rule: no
    review of the reasoning catches it, the gate does, regardless of
    how the agent was corrupted upstream."""
    d = gate_rule_code(
        "quartic", INVERTED_RULE, theorem_fallback("quartic", 200),
        m=800, n_coeffs=200, epsilon=0.02, delta=0.05,
    )
    assert d.released == "fallback"
    # released risk equals fallback risk: bound holds trivially
    assert d.candidate_holdout_accuracy < 0.5


def test_gate_end_to_end_crash_fails_closed():
    d = gate_rule_code(
        "quartic", CRASHING_RULE, theorem_fallback("quartic", 200),
        m=400, n_coeffs=200,
    )
    assert d.released == "fallback"


def test_decision_is_auditable():
    y = np.zeros(100, dtype=int)
    d = certify_release(np.zeros(100, dtype=int), np.zeros(100, dtype=int), y)
    out = d.to_dict()
    for key in ("released", "margin", "epsilon", "delta_family",
                "delta_per_candidate", "n_candidates", "fallback_kind",
                "certificate"):
        assert key in out
