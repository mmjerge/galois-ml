"""Final-artifact release gate: corruption-independent risk control
relative to a trusted fallback.

This implements the top-priority proposition from the external design
review (GPT-6 Astra, 2026-09-09, "Proposition 6"), which answers the
pilot's central lesson: detection and truthful warnings need not
produce mitigation, so the mitigation decision is enforced OUTSIDE the
agent. The agent's rule is executable; its accuracy is checkable
against fresh data regardless of how the agent reasoned or how corrupt
its evidence was.

PROPOSITION (fresh-holdout gate). Hypotheses:

  1. the agent proposes a frozen executable rule h; the trusted
     fallback h0 is fixed;
  2. both are fixed independently of a fresh holdout
     H = ((X_i, Y_i))_{i=1..m} ~ D^m -- the candidate may otherwise
     depend arbitrarily on corrupted evidence;
  3. labels and evaluator are trusted; evaluation is sandboxed and
     failures release h0;
  4. the certification set is not reused for adaptive proposal
     selection (for K candidates all fixed independently of H, use
     delta/K).

Define the paired loss difference
  D_hat = (1/m) sum_i [ 1{h(X_i) != Y_i} - 1{h0(X_i) != Y_i} ]
and the margin r_m(delta) = sqrt(2 log(1/delta) / m).

Release h iff D_hat + r_m(delta) <= epsilon; otherwise release h0.
Then for the released rule h_out:

  Pr[ L_D(h_out) > L_D(h0) + epsilon ] <= delta.

Proof: each paired difference lies in [-1, 1]; one-sided Hoeffding
gives Pr[ (L_D(h) - L_D(h0)) - D_hat > r_m(delta) ] <= delta;
on the complement, acceptance implies the bound, and rejection releases
h0 itself. Conditioning on all proposal-generation randomness handles a
random candidate independent of H.

What this does and does not give (review's own caveats, kept verbatim
in spirit):

* It holds against ARBITRARY upstream tool-response corruption,
  including consistent label-permutation attacks -- the class no
  evidence-level verifier can catch.
* It is fallback-relative, not clean-agent-relative: an attacker can
  force fallback on every run, destroying the agent's improvement over
  h0. Bounding degradation against the CLEAN agent needs a shadow run
  or an independent bound on the fallback's gap.
* The population claim is statistical and assumes the deployment law
  matches the certification law. On this testbed that law is the
  synthetic generator, so fresh holdouts are genuinely IID by
  construction -- the one setting where hypothesis 2 is checkable
  rather than hoped for.
* The fallback choice sets the floor. A theorem-based fallback encodes
  task knowledge in the trusted evaluator; that is sensible for
  deployment and changes the scientific claim, so record which fallback
  a number was certified against.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import log, sqrt
from typing import Callable

import numpy as np


def hoeffding_margin(m: int, delta: float, n_candidates: int = 1) -> float:
    """sqrt(2 log(K/delta) / m) for paired losses in [-1, 1].

    Computed as log(K) - log(delta) for numerical stability at small
    failure budgets, rather than via 1/(delta/K).
    """
    if not isinstance(m, (int, np.integer)) or m <= 0:
        raise ValueError("holdout size m must be a positive integer")
    if not (np.isfinite(delta) and 0 < delta < 1):
        raise ValueError("delta must be finite and in (0, 1)")
    if not isinstance(n_candidates, (int, np.integer)) or n_candidates < 1:
        raise ValueError("n_candidates must be a positive integer")
    return sqrt(2.0 * (log(n_candidates) - log(delta)) / m)


@dataclass
class GateDecision:
    """The release decision and everything needed to audit it."""

    released: str                    # 'candidate' | 'fallback'
    reason: str
    m: int
    epsilon: float
    delta_family: float              # failure budget for the whole
                                     # K-candidate certification family
    delta_per_candidate: float       # delta_family / n_candidates; this
                                     # is NOT the failure probability of
                                     # a post-holdout selection among
                                     # gated candidates -- that is
                                     # bounded by delta_family
    n_candidates: int                # declared bound on ALL candidates
                                     # examined against this holdout,
                                     # not just those reported
    margin: float                    # sqrt(2 log(K/delta)/m)
    paired_loss_difference: float    # D_hat (negative = candidate better)
    candidate_holdout_accuracy: float | None
    fallback_holdout_accuracy: float
    fallback_kind: str | None        # which fallback was actually used
    certificate: str                 # the inequality that was checked

    def to_dict(self) -> dict:
        return asdict(self)


def certify_release(
    candidate_preds: np.ndarray | None,
    fallback_preds: np.ndarray,
    y: np.ndarray,
    epsilon: float = 0.02,
    delta: float = 0.05,
    n_candidates: int = 1,
    fallback_kind: str | None = None,
) -> GateDecision:
    """Apply the release rule to predictions on a fresh holdout.

    SCOPE (external review, sec 3.3): this is a GATE-DECISION
    implementation. It computes and certifies the decision; it does not
    by itself prove the running pipeline enforces gated release. The
    outer trusted runtime must freeze/hash the artifact, evaluate that
    exact artifact, atomically select, prevent post-certification
    substitution, and define trusted timeout/crash behavior. Exceptions
    raised here are fail-safe only if that runtime releases nothing on
    exception.

    Args:
        candidate_preds: candidate predictions on the holdout, or None
            if the candidate failed to execute (failures release the
            fallback).
        fallback_preds: trusted fallback's predictions on the holdout.
        y: trusted labels.
        epsilon: allowed excess risk over the fallback; must be in
            [0, 1] (negative epsilon voids the theorem).
        delta: FAMILY failure budget. `n_candidates` must upper-bound
            ALL candidates examined against this holdout, not just the
            ones reported; the per-candidate bound is delta/n_candidates
            and the family-level guarantee is at delta. This correction
            covers a fixed batch of independently-frozen candidates; it
            does NOT license adaptive proposal selection using feedback
            from the same holdout.
        fallback_kind: recorded provenance of the fallback used.
    """
    y = np.asarray(y)
    if y.ndim != 1 or len(y) == 0:
        raise ValueError("labels must be a nonempty 1-d array")
    fallback_preds = np.asarray(fallback_preds)
    if fallback_preds.shape != y.shape:
        raise ValueError("fallback predictions must match labels")
    # epsilon < 0 is a theorem bug, not a stricter gate: releasing the
    # fallback itself then violates L(out) <= L(h0) + epsilon, so the
    # claimed failure probability can be 1 (external review, sec 3.4).
    if not (np.isfinite(epsilon) and 0.0 <= epsilon <= 1.0):
        raise ValueError("epsilon must be finite and in [0, 1]")
    m = len(y)
    margin = hoeffding_margin(m, delta, n_candidates)  # validates inputs
    fb_acc = float(np.mean(fallback_preds == y))
    common = dict(
        m=m, epsilon=epsilon, delta_family=delta,
        delta_per_candidate=delta / n_candidates,
        n_candidates=n_candidates, margin=margin,
        fallback_holdout_accuracy=fb_acc, fallback_kind=fallback_kind,
    )

    if candidate_preds is None:
        return GateDecision(
            released="fallback",
            reason="candidate failed to execute; gate fails closed",
            paired_loss_difference=float("nan"),
            candidate_holdout_accuracy=None,
            certificate="released fallback: L(out) = L(h0) trivially",
            **common,
        )

    candidate_preds = np.asarray(candidate_preds)
    if candidate_preds.shape != y.shape:
        raise ValueError("candidate predictions must match labels")
    cand_loss = (candidate_preds != y).astype(float)
    fb_loss = (fallback_preds != y).astype(float)
    d_hat = float(np.mean(cand_loss - fb_loss))

    accept = d_hat + margin <= epsilon
    return GateDecision(
        released="candidate" if accept else "fallback",
        reason=(
            f"D_hat + r = {d_hat:+.4f} + {margin:.4f} "
            f"{'<=' if accept else '>'} epsilon = {epsilon:.4f}"
        ),
        paired_loss_difference=d_hat,
        candidate_holdout_accuracy=float(np.mean(candidate_preds == y)),
        certificate=(
            f"family-level: Pr[L(released) > L(fallback) + {epsilon}] "
            f"<= {delta} over all {n_candidates} candidate(s) certified "
            f"against this holdout"
        ),
        **common,
    )


def gate_rule_code(
    task: str,
    candidate_code: str,
    fallback_predict: Callable[[np.ndarray], np.ndarray],
    m: int = 2000,
    n_coeffs: int = 1000,
    holdout_seed: int | None = 900_001,
    epsilon: float = 0.02,
    delta: float = 0.05,
    n_candidates: int = 1,
) -> GateDecision:
    """Certify an agent-produced `predict(zeta)` rule on a FRESH holdout.

    The holdout is generated from the task's law at `holdout_seed`.
    Hypothesis 2 is the caller's obligation, and (external review, sec
    3.5) a FIXED PUBLIC seed is not an adversarial freshness mechanism:
    an attacker who knows the code and generator can specialize a
    candidate to a reproducible holdout. Pass `holdout_seed=None` to
    draw the seed from OS entropy at certification time (it is recorded
    in the decision's reason field for auditors); keep it inaccessible
    to candidate execution; and freeze the candidate BEFORE the seed is
    drawn. The default fixed seed remains only for reproducible tests
    and demos. A PRNG draw is an engineering approximation of the IID
    hypothesis, not a proof of it; and `run_predict_code`'s isolation
    is an assumption of this function, established elsewhere.

    Any exception during candidate execution releases the fallback
    (fail closed at this layer; see certify_release for the outer
    runtime's obligations).
    """
    import secrets

    from galois_ml.data import make_dataset
    from galois_ml.llm_eval import run_predict_code

    if holdout_seed is None:
        holdout_seed = secrets.randbits(63)
    X, y, _ = make_dataset(task, m, n_coeffs, seed=holdout_seed)
    fallback_preds = np.asarray(fallback_predict(X))
    try:
        candidate_preds = run_predict_code(candidate_code, X)
    except Exception:
        candidate_preds = None
    decision = certify_release(
        candidate_preds, fallback_preds, y,
        epsilon=epsilon, delta=delta, n_candidates=n_candidates,
        fallback_kind=getattr(fallback_predict, "fallback_kind", None),
    )
    decision.reason += f" [holdout_seed={holdout_seed}]"
    return decision


def theorem_fallback(task: str, n_coeffs: int = 1000):
    """A trusted fallback from the proven criteria where implemented,
    else the Chebotarev Bayes rule. NOTE: both encode task knowledge in
    the trusted evaluator -- legitimate for deployment, but a certified
    number is fallback-relative, so report which fallback was used."""
    from galois_ml.criteria import ChebotarevBayesClassifier, TheoremClassifier

    try:
        clf = TheoremClassifier(task, n_coeffs)
        kind = "theorem_criteria"
    except ValueError:
        clf = ChebotarevBayesClassifier(task, n_coeffs)
        kind = "chebotarev_bayes"
    def predict(X, _clf=clf):
        return _clf.predict(X)

    predict.fallback_kind = kind  # recorded by gate_rule_code
    return predict
