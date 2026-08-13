"""Correlated failure across independent agents: does redundancy buy
independence?

Every multi-agent assurance argument -- verification, voting, Byzantine
fault tolerance -- rests on an unstated premise: that independently
running agents fail *independently*. If k agents share a model and a
prompt, they may instead fail in lockstep, in which case k reviewers
provide one check, not k, and a majority vote inherits the single
agent's blind spots rather than cancelling them.

Anthropic's Frontier Red Team report on emerging multiagent systems
(August 2026) documents the qualitative version of this: agents are
"low variance," and "when one agent makes a bad decision, it is likely
that many agents will make that same bad decision," turning isolated
errors into systemic failures. Their examples are striking (18 of 30
agents choosing the identical git branch name; agents in a writing
workshop independently producing the same title).

This module measures the quantitative version on a task with provable
ground truth, which lets us compute what that correlation costs:

* per-agent accuracy and regret against the exact Bayes ceiling;
* pairwise error overlap (Jaccard index on the sets of misclassified
  test fields) against the overlap expected if failures were
  independent, given each agent's error rate;
* majority-vote accuracy, versus the accuracy an ensemble of genuinely
  independent agents with the same individual error rates would reach;
* effective independent count: the number of truly independent voters
  the observed correlation is worth. If k agents behave as ~1, the
  redundancy is illusory.

The point is not that agents are bad. It is that "we added reviewers"
is an assurance claim that can be checked, and here it can be checked
exactly.
"""

from __future__ import annotations

import itertools
from dataclasses import asdict, dataclass, field

import numpy as np

from galois_ml.benchmark import GaloisBenchmark


@dataclass
class AgentRun:
    """One agent's proposal, scored against ground truth."""

    seed: int
    valid: bool
    accuracy: float | None = None
    conjecture: str | None = None
    code: str | None = None
    # boolean mask over the test split: True where this agent was wrong
    error_mask: list[bool] = field(default_factory=list)
    error: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ConformityResult:
    task: str
    model_id: str
    n_agents: int
    bayes_accuracy: float
    runs: list[dict] = field(default_factory=list)
    metrics: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


# -- correlation analysis ---------------------------------------------------

def pairwise_error_overlap(masks: list[np.ndarray]) -> dict:
    """Observed vs independent-baseline overlap of error sets.

    For two agents with error rates p1, p2, independent failures would
    intersect on a p1*p2 fraction of items. The Jaccard index of the
    observed error sets, compared against the Jaccard implied by
    independence, says whether the agents are failing on the *same*
    items or merely at the same rate.

    Returns means over all pairs, plus the ratio observed/independent
    (1.0 means independent; >1 means correlated failure).
    """
    observed, expected = [], []
    for a, b in itertools.combinations(masks, 2):
        n = len(a)
        p1, p2 = a.mean(), b.mean()
        inter = float(np.logical_and(a, b).sum())
        union = float(np.logical_or(a, b).sum())
        observed.append(inter / union if union > 0 else 0.0)
        # independence: E|A^B| = n*p1*p2, E|AuB| = n*(p1 + p2 - p1*p2)
        exp_inter = n * p1 * p2
        exp_union = n * (p1 + p2 - p1 * p2)
        expected.append(exp_inter / exp_union if exp_union > 0 else 0.0)
    if not observed:
        return {"observed_jaccard": None, "independent_jaccard": None,
                "correlation_ratio": None, "n_pairs": 0}
    obs_mean = float(np.mean(observed))
    exp_mean = float(np.mean(expected))
    return {
        "observed_jaccard": obs_mean,
        "independent_jaccard": exp_mean,
        "correlation_ratio": (obs_mean / exp_mean) if exp_mean > 0 else None,
        "n_pairs": len(observed),
    }


def majority_vote_accuracy(
    predictions: list[np.ndarray], y_true: np.ndarray
) -> float:
    """Accuracy of a plurality vote over agent predictions (ties broken
    toward the lowest label index, deterministically)."""
    stacked = np.stack(predictions)  # (n_agents, n_items)
    n_classes = int(max(stacked.max(), y_true.max())) + 1
    counts = np.stack(
        [(stacked == c).sum(axis=0) for c in range(n_classes)]
    )  # (n_classes, n_items)
    voted = counts.argmax(axis=0)
    return float(np.mean(voted == y_true))


def independent_ensemble_accuracy(
    error_rates: list[float], n_items: int = 200_000, seed: int = 0
) -> float:
    """Majority-vote accuracy of hypothetical agents with the given
    individual error rates whose errors are independent.

    This is the counterfactual the correlated ensemble is compared to:
    the assurance a system designer implicitly assumes when adding
    reviewers. Binary tasks only (errors flip the label).
    """
    rng = np.random.default_rng(seed)
    wrong = np.stack([
        rng.random(n_items) < rate for rate in error_rates
    ])  # True where that agent errs
    # with binary labels, a majority of wrong votes flips the outcome
    majority_wrong = wrong.sum(axis=0) > (len(error_rates) / 2)
    return float(1.0 - majority_wrong.mean())


def effective_independent_count(
    observed_majority_acc: float,
    error_rates: list[float],
    max_k: int | None = None,
    seed: int = 0,
) -> float:
    """How many genuinely independent agents the observed ensemble is
    worth.

    We find the smallest k such that an independent ensemble of k agents
    (each at the mean observed error rate) matches or exceeds the
    observed majority-vote accuracy. Reported as a float via linear
    interpolation between k and k+1. A value near 1 with k agents
    deployed means the redundancy bought essentially nothing.
    """
    mean_rate = float(np.mean(error_rates))
    max_k = max_k or (2 * len(error_rates) + 1)
    accs = {}
    for k in range(1, max_k + 1, 2):  # odd k avoids tie handling
        accs[k] = independent_ensemble_accuracy([mean_rate] * k, seed=seed)
    ks = sorted(accs)
    if observed_majority_acc <= accs[ks[0]]:
        return 1.0
    for lo, hi in zip(ks, ks[1:]):
        if accs[lo] <= observed_majority_acc <= accs[hi]:
            span = accs[hi] - accs[lo]
            frac = 0.0 if span == 0 else (observed_majority_acc - accs[lo]) / span
            return float(lo + frac * (hi - lo))
    return float(ks[-1])


def analyze_runs(
    runs: list[AgentRun], bench: GaloisBenchmark, predictions: list[np.ndarray]
) -> dict:
    """Assemble the correlated-failure metrics for a set of agent runs."""
    valid = [(r, p) for r, p in zip(runs, predictions) if r.valid]
    if len(valid) < 2:
        return {"note": "fewer than two valid runs; no correlation analysis"}
    masks = [np.array(r.error_mask, dtype=bool) for r, _ in valid]
    preds = [p for _, p in valid]
    rates = [float(m.mean()) for m in masks]
    accs = [float(r.accuracy) for r, _ in valid]

    overlap = pairwise_error_overlap(masks)
    observed_majority = majority_vote_accuracy(preds, bench.y_test)
    independent_majority = independent_ensemble_accuracy(rates)
    metrics = {
        "n_valid_agents": len(valid),
        "individual_accuracy_mean": float(np.mean(accs)),
        "individual_accuracy_std": float(np.std(accs, ddof=1)) if len(accs) > 1 else 0.0,
        "individual_accuracy_min": float(np.min(accs)),
        "individual_accuracy_max": float(np.max(accs)),
        "bayes_accuracy": bench.bayes_accuracy,
        "mean_regret": float(bench.bayes_accuracy - np.mean(accs)),
        **overlap,
        "majority_vote_accuracy": observed_majority,
        "majority_vote_regret": bench.bayes_accuracy - observed_majority,
        "independent_ensemble_accuracy": independent_majority,
        # how much of the theoretical ensembling gain was actually realized
        "ensemble_gain_observed": observed_majority - float(np.mean(accs)),
        "ensemble_gain_if_independent": independent_majority - float(np.mean(accs)),
        "effective_independent_agents": effective_independent_count(
            observed_majority, rates
        ),
        # do agents state the same rule? crude but informative
        "distinct_conjectures": len({
            (r.conjecture or "").strip().lower()[:200] for r, _ in valid
        }),
    }
    gain_if_ind = metrics["ensemble_gain_if_independent"]
    metrics["realized_gain_fraction"] = (
        float(metrics["ensemble_gain_observed"] / gain_if_ind)
        if gain_if_ind > 1e-9 else None
    )
    return metrics


def run_conformity_experiment(
    bench: GaloisBenchmark,
    model_id: str,
    n_agents: int = 5,
    region: str = "us-east-1",
    temperature: float = 1.0,
    theory_in_brief: bool = True,
) -> ConformityResult:
    """Run n independent analyst agents on the same task and measure
    whether their failures are independent.

    Temperature defaults to 1.0: we are deliberately giving the agents
    every chance to diverge. Correlated failure at high temperature is
    a stronger result than correlated failure at temperature 0.
    """
    from galois_ml.llm_eval import extract_json, run_predict_code
    from galois_ml.multi_agent import (
        ANALYST_ROLE,
        _make_agent,
        _task_brief,
        build_tools,
    )

    result = ConformityResult(
        task=bench.task,
        model_id=model_id,
        n_agents=n_agents,
        bayes_accuracy=bench.bayes_accuracy,
    )
    brief = _task_brief(bench)
    if not theory_in_brief:
        from galois_ml.tool_poisoning import BRIEF_WITHOUT_THEORY_NOTE

        head, _, tail = brief.partition("Element order counts")
        _, _, rest = tail.partition("\n\nFacts:")
        brief = head + BRIEF_WITHOUT_THEORY_NOTE + "\n\nFacts:" + rest

    runs: list[AgentRun] = []
    predictions: list[np.ndarray] = []
    for seed in range(n_agents):
        run = AgentRun(seed=seed, valid=False)
        try:
            agent = _make_agent(
                model_id, region, ANALYST_ROLE, build_tools(bench),
                temperature=temperature,
            )
            text = str(agent(brief))
            parsed = extract_json(text)
            preds = run_predict_code(parsed["python"], bench.X_test)
            run.accuracy = float(np.mean(preds == bench.y_test))
            run.error_mask = (preds != bench.y_test).tolist()
            run.conjecture = parsed.get("conjecture")
            run.code = (parsed.get("python") or "")[:2000]
            run.valid = True
            predictions.append(preds)
        except Exception as exc:
            run.error = f"{type(exc).__name__}: {exc}"
            predictions.append(np.full(len(bench.y_test), -1))
        runs.append(run)

    result.runs = [r.to_dict() for r in runs]
    result.metrics = analyze_runs(runs, bench, predictions)
    return result
