"""Tests for the trusted release runtime (galois_ml.release).

Each test discharges one of the six review obligations, plus the
isolation property (labels never enter the candidate process) at the
design level and the fail-closed matrix for every evaluation failure
mode.
"""

import numpy as np
import pytest

from galois_ml.gate import theorem_fallback
from galois_ml.release import ReleaseRuntime, _sha256

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
INVERTED_RULE = GOOD_RULE.replace("return 0", "return TMP").replace(
    "return 1", "return 0"
).replace("return TMP", "return 1")
CRASHING_RULE = "def predict(zeta):\n    raise RuntimeError('boom')\n"
HANGING_RULE = (
    "def predict(zeta):\n"
    "    while True:\n"
    "        pass\n"
)
BAD_OUTPUT_RULE = "def predict(zeta):\n    return float('nan')\n"


def runtime(**kw):
    # epsilon must exceed the m=300 margin (0.141 at delta=0.05, up to
    # 0.165 at K=3) for an approximately-equal candidate to be
    # releasable: the gate refuses to certify beyond what the sample
    # size resolves, so small-m tests need a loose epsilon
    args = dict(m=300, n_coeffs=200, epsilon=0.20, delta=0.05,
                eval_timeout_s=30.0)
    args.update(kw)
    return ReleaseRuntime(
        "quartic", theorem_fallback("quartic", args["n_coeffs"]), **args
    )


# -- obligations 1, 2, 4: freeze, hash, exact artifact, no substitution ----

def test_happy_path_releases_candidate_with_hash():
    rt = runtime()
    h = rt.submit(GOOD_RULE)
    assert h == _sha256(GOOD_RULE)
    assert rt.freeze() == [h]
    decisions = rt.certify()
    assert len(decisions) == 1
    art = rt.release()
    assert art.kind == "candidate"
    assert art.artifact_hash == h
    # the released predictor is rebuilt from the certified bytes and works
    from galois_ml.data import make_dataset

    X, y, _ = make_dataset("quartic", 100, 200, seed=123)
    assert np.mean(np.asarray(art.predict(X)) == y) > 0.9


def test_no_submission_after_freeze():
    rt = runtime()
    rt.submit(GOOD_RULE)
    rt.freeze()
    with pytest.raises(RuntimeError, match="frozen"):
        rt.submit(INVERTED_RULE)


def test_lifecycle_order_enforced():
    rt = runtime()
    with pytest.raises(RuntimeError):
        rt.freeze()                       # nothing submitted
    rt.submit(GOOD_RULE)
    with pytest.raises(RuntimeError):
        rt.certify()                      # not frozen
    with pytest.raises(RuntimeError):
        rt.release()                      # not certified
    rt.freeze()
    with pytest.raises(RuntimeError):
        rt.freeze()                       # double freeze
    rt.certify()
    with pytest.raises(RuntimeError):
        rt.certify()                      # holdout reuse
    rt.release()
    with pytest.raises(RuntimeError):
        rt.release()                      # double release


# -- obligation 3: atomic selection; gate actually decides ------------------

def test_inverted_candidate_yields_fallback():
    rt = runtime()
    rt.submit(INVERTED_RULE)
    rt.freeze()
    rt.certify()
    art = rt.release()
    assert art.kind == "fallback"
    assert art.artifact_hash is None
    assert art.decision.released == "fallback"


def test_multi_candidate_selection_and_family_budget():
    rt = runtime()
    rt.submit(INVERTED_RULE)
    rt.submit(GOOD_RULE)
    rt.submit(CRASHING_RULE)
    rt.freeze()
    decisions = rt.certify()
    # every decision was certified at the family-corrected budget
    assert all(d.n_candidates == 3 for d in decisions)
    assert all(d.delta_per_candidate == pytest.approx(0.05 / 3)
               for d in decisions)
    art = rt.release()
    assert art.kind == "candidate"
    assert art.artifact_hash == _sha256(GOOD_RULE)


# -- obligation 6: fail-closed matrix ---------------------------------------

@pytest.mark.parametrize("code,reason_fragment", [
    (CRASHING_RULE, "candidate raised"),
    # NaN output crashes inside run_predict_code's own int conversion,
    # so it surfaces as a raise; the runtime's finiteness check is the
    # second line of defense for evaluators that pass NaN through
    (BAD_OUTPUT_RULE, "candidate raised"),
])
def test_evaluation_failures_release_fallback(code, reason_fragment):
    rt = runtime()
    rt.submit(code)
    rt.freeze()
    rt.certify()
    art = rt.release()
    assert art.kind == "fallback"
    assert any(reason_fragment in n for n in art.audit["evaluation_notes"])


def test_hanging_candidate_times_out_to_fallback():
    rt = runtime(eval_timeout_s=3.0)
    rt.submit(HANGING_RULE)
    rt.freeze()
    rt.certify()
    art = rt.release()
    assert art.kind == "fallback"
    assert "timeout" in art.audit["evaluation_notes"][0]


# -- holdout secrecy and audit -----------------------------------------------

def test_seed_is_entropy_drawn_and_disclosed_only_in_audit():
    rt = runtime()
    rt.submit(GOOD_RULE)
    rt.freeze()
    rt.certify()
    # no public attribute exposes the seed pre-release
    public = [a for a in dir(rt) if not a.startswith("_")]
    assert public == sorted(
        ["submit", "freeze", "certify", "release"]
    ), f"unexpected public surface: {public}"
    art = rt.release()
    assert isinstance(art.audit["holdout_seed"], int)
    # two runtimes draw different holdouts: no reproducible-holdout
    # specialization across releases
    rt2 = runtime()
    rt2.submit(GOOD_RULE)
    rt2.freeze()
    rt2.certify()
    art2 = rt2.release()
    assert art2.audit["holdout_seed"] != art.audit["holdout_seed"]


def test_audit_record_is_complete():
    rt = runtime()
    rt.submit(GOOD_RULE)
    rt.freeze()
    rt.certify()
    art = rt.release()
    for key in ("task", "holdout_seed", "holdout_m", "epsilon",
                "delta_family", "n_candidates", "candidate_hashes",
                "evaluation_notes", "decisions", "released_at"):
        assert key in art.audit
    assert art.audit["decisions"][0]["certificate"].startswith("family-level")


# -- isolation: labels never enter the candidate process ---------------------

def test_candidate_worker_receives_features_only():
    """Design-level check: the worker signature admits code, features,
    and a pipe -- there is no parameter through which labels or the
    seed could be passed, so leaking them requires editing the trusted
    module, not misusing it."""
    import inspect

    from galois_ml.release import _candidate_worker

    params = list(inspect.signature(_candidate_worker).parameters)
    assert params == ["code", "X", "conn"]


# -- fourth-review regressions ------------------------------------------------

def test_mutation_of_returned_decisions_cannot_alter_release():
    """Review counterexample (verified before fixing): flipping fields
    on the decisions returned by certify() flipped the release. The
    authorization must be private and immutable."""
    rt = runtime()
    rt.submit("def predict(zeta):\n    return 1\n")  # constant, refused
    rt.freeze()
    decisions = rt.certify()
    assert decisions[0].released == "fallback"
    decisions[0].released = "candidate"           # the attack
    decisions[0].paired_loss_difference = -1.0
    art = rt.release()
    assert art.kind == "fallback"                 # unaffected
    # post-release mutation of the artifact's copies is equally inert
    art.audit["released"] = "candidate"
    assert art.kind == "fallback"


def test_certification_failure_is_terminal():
    """A failing trusted component must leave the runtime FAILED, not
    FROZEN-with-partial-decisions: a retry would mix holdouts."""
    def broken_fallback(X):
        raise RuntimeError("fallback infrastructure down")

    rt = ReleaseRuntime("quartic", broken_fallback, m=200, n_coeffs=200,
                        epsilon=0.20, eval_timeout_s=30.0)
    rt.submit(GOOD_RULE)
    rt.freeze()
    with pytest.raises(RuntimeError):
        rt.certify()
    with pytest.raises(RuntimeError, match="failed"):
        rt.certify()                              # no retry
    with pytest.raises(RuntimeError, match="failed"):
        rt.release()                              # no release either


def test_out_of_alphabet_labels_fail_closed():
    rt = runtime()
    rt.submit("def predict(zeta):\n    return 7\n")  # not a valid label
    rt.freeze()
    rt.certify()
    art = rt.release()
    assert art.kind == "fallback"
    assert any("alphabet" in n for n in art.audit["evaluation_notes"])


def test_invalid_fallback_predictions_refuse_certification():
    def bad_fallback(X):
        import numpy as np
        return np.full(len(X), 9, dtype=int)      # outside alphabet

    rt = ReleaseRuntime("quartic", bad_fallback, m=200, n_coeffs=200,
                        epsilon=0.20, eval_timeout_s=30.0)
    rt.submit(GOOD_RULE)
    rt.freeze()
    with pytest.raises(RuntimeError, match="trusted-component failure"):
        rt.certify()


def test_released_candidate_executes_through_isolated_path():
    """Deployment must use the same constrained protocol as
    certification: spawned process, byte protocol, timeout. A released
    hanging rule therefore raises instead of hanging the caller."""
    rt = runtime()
    rt.submit(GOOD_RULE)
    rt.freeze()
    rt.certify()
    art = rt.release()
    assert art.kind == "candidate"
    from galois_ml.data import make_dataset

    X, y, _ = make_dataset("quartic", 60, 200, seed=77)
    preds = art.predict(X)
    assert np.mean(np.asarray(preds) == y) > 0.9
    # the deployment path enforces the timeout too
    from galois_ml.release import isolated_predict

    with pytest.raises(RuntimeError, match="timeout"):
        isolated_predict(HANGING_RULE, X[:5], timeout_s=3.0)


def test_config_validated_before_state_exists():
    fb = theorem_fallback("quartic", 200)
    with pytest.raises(ValueError):
        ReleaseRuntime("quartic", fb, epsilon=-0.1)
    with pytest.raises(ValueError):
        ReleaseRuntime("quartic", fb, delta=1.5)
    with pytest.raises(ValueError):
        ReleaseRuntime("quartic", fb, m=0)
