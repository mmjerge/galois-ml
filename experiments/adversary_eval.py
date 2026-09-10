"""Adaptive-adversary experiments. Offline: no model API, no credentials.

Closes the two evidential gaps flagged in the calibrated assessment and
by the fourth review:

PART 1 -- AN IMPLEMENTED ADAPTIVE ATTACKER, not a theorized one.
`data_adaptive` plays the optimal strategy against the evidence layer:
one consistent label permutation across distributions and examples,
exact honest grammar, attainable values, matching class names, honest
validation (any validation lie is caught by recomputation). We verify:

  a. the guard, with every check enabled, detects 0 of N poisoned
     responses (deterministic -- the restricted impossibility result
     made concrete in running code);
  b. a credulous learner trained on the permuted evidence produces an
     inverted classifier whose true accuracy is near 1 - ceiling;
  c. the release gate refuses that classifier, so the released-risk
     bound holds against the attack the evidence layer provably cannot
     stop. This is the full three-layer story executed end to end
     against the strongest channel attacker in the model.

PART 2 -- GATE OPERATING CHARACTERISTICS with an independent audit.
The 0/178 count in the archived study is an algebraic property of the
decision rule. Here we do what the review asked instead: designed
boundary candidates with KNOWN population excess risk Delta (fallback
predictions flipped by a deterministic per-row hash rule at rate q, so
Delta ~= q * (2*acc_fb - 1), frozen before any holdout exists), swept
across the acceptance boundary, each certified on a fresh holdout, and
every RELEASED artifact audited on a large INDEPENDENT sample. That
yields real acceptance-probability curves and a non-tautological
estimate of the population-bound violation rate. certify_release is
called directly (the statistics of the decision rule are under test,
not the process isolation, which has its own tests).

Run:  python experiments/adversary_eval.py [--reps 30] [--m 2000]
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parent.parent))

from galois_ml.benchmark import load_benchmark
from galois_ml.data import make_dataset
from galois_ml.gate import certify_release, theorem_fallback
from galois_ml.groups import GROUPS
from galois_ml.guard import ToolGuard
from galois_ml.multi_agent import (
    coefficient_distribution,
    labeled_examples,
    validate_rule,
)
from galois_ml.tool_poisoning import build_poisoned_tools

OUT = Path("experiments/outputs")
TASK = "sextic"
N_COEFFS = 200  # fast; the attack and the gate are indifferent to width


# -- part 1: the adaptive attacker end to end --------------------------------

def adaptive_attacker_vs_guard(bench) -> dict:
    """Exercise every tool on the data_adaptive channel through a fully
    armed guard; detection is deterministic, so counts are exact."""
    guard = ToolGuard(
        degree=GROUPS[bench.class_names[0]].order,
        n_coeffs=bench.X_train.shape[1],
        class_names=bench.class_names,
        recompute_validation=lambda c: validate_rule(bench, c),
        recompute_distribution=lambda n: coefficient_distribution(bench, n),
        recompute_examples=lambda lab: labeled_examples(bench, lab),
        policy="repair",
    )
    counter: dict = {}
    tools = build_poisoned_tools(bench, "data_adaptive", counter, guard=guard)

    def call(name, **kw):
        for t in tools:
            if getattr(t, "tool_name", getattr(t, "__name__", "")) == name:
                return str(t(**kw) if kw else t())
        raise AssertionError(name)

    for n in (2, 4, 8, 9, 25, 27, 49):
        call("inspect_coefficient", n=n)
    call("list_prime_powers")
    for label in range(len(bench.class_names)):
        call("show_examples", label=label)
    call("test_rule", code="def predict(zeta):\n    return 0\n")
    return {
        "poisoned_calls": counter.get("poisoned_calls", 0),
        "guard_violations": len(guard.violations),
        "guard_detected": guard.detected,
    }


def credulous_learner_under_attack(bench, seed: int = 0) -> dict:
    """A learner that trusts the evidence channels (and does not or
    cannot use validation) sees exactly the permuted world: train on
    honest features with permuted labels. Its true accuracy lands near
    1 - ceiling: the confidently-wrong inversion, produced by an
    attacker the guard provably cannot flag."""
    from sklearn.tree import DecisionTreeClassifier

    n_classes = len(bench.class_names)
    y_permuted = (bench.y_train + 1) % n_classes
    model = DecisionTreeClassifier(random_state=seed).fit(
        bench.X_train, y_permuted
    )
    return {
        "true_accuracy": float(np.mean(
            model.predict(bench.X_test) == bench.y_test
        )),
        "accuracy_in_attackers_world": float(np.mean(
            model.predict(bench.X_test) == (bench.y_test + 1) % n_classes
        )),
        "model": model,
    }


def gate_the_attacked_learner(model, epsilon, delta, m, seed) -> dict:
    """The last line: gate the inverted classifier against the theorem
    fallback on a fresh holdout, then audit the released artifact
    independently."""
    fb = theorem_fallback(TASK, N_COEFFS)
    X, y, _ = make_dataset(TASK, m, N_COEFFS, seed=seed)
    d = certify_release(model.predict(X), fb(X), y,
                        epsilon=epsilon, delta=delta,
                        fallback_kind=fb.fallback_kind)
    Xa, ya, _ = make_dataset(TASK, 20_000, N_COEFFS, seed=seed + 1)
    released_preds = model.predict(Xa) if d.released == "candidate" else fb(Xa)
    audit_acc = float(np.mean(released_preds == ya))
    fb_audit_acc = float(np.mean(fb(Xa) == ya))
    return {
        "released": d.released,
        "candidate_holdout_accuracy": d.candidate_holdout_accuracy,
        "released_audit_accuracy": audit_acc,
        "fallback_audit_accuracy": fb_audit_acc,
        "released_excess_on_independent_audit": (
            (1 - audit_acc) - (1 - fb_audit_acc)
        ),
    }


# -- part 2: gate operating characteristics ----------------------------------

def _flip_rows(X: np.ndarray, q: float) -> np.ndarray:
    """Deterministic pointwise row selector at rate ~q: hash the row
    bytes. Fixed before any holdout exists; no cross-row state."""
    thresh = int(q * 2**32)
    out = np.empty(len(X), dtype=bool)
    for i, row in enumerate(X):
        h = hashlib.sha256(row.tobytes()).digest()
        out[i] = int.from_bytes(h[:4], "big") < thresh
    return out


def make_boundary_candidate(q: float):
    """Candidate = fallback's prediction, flipped on the hash-selected
    rows. Population excess risk ~= q * (2*acc_fb - 1); q sweeps it
    across the acceptance boundary."""
    fb = theorem_fallback(TASK, N_COEFFS)

    def predict(X, _fb=fb, _q=q):
        preds = np.asarray(_fb(X)).copy()
        flip = _flip_rows(np.asarray(X, dtype=float), _q)
        preds[flip] = 1 - preds[flip]
        return preds

    return predict


def operating_characteristics(args) -> list[dict]:
    fb = theorem_fallback(TASK, N_COEFFS)
    Xa, ya, _ = make_dataset(TASK, 100_000, N_COEFFS, seed=777_001)
    fb_a = fb(Xa)
    acc_fb = float(np.mean(fb_a == ya))
    rows = []
    rng = np.random.default_rng(20260909)
    print(f"\nPART 2: operating characteristics "
          f"(fallback audit acc {acc_fb:.4f}; epsilon={args.epsilon}, "
          f"delta={args.delta}, reps={args.reps})")
    print(f"{'q':>6}{'true Delta':>12}{'m':>8}{'accept rate':>13}"
          f"{'violations/released':>21}")
    for q in args.q_grid:
        cand = make_boundary_candidate(q)
        cand_a = cand(Xa)
        true_delta = float(np.mean(cand_a != ya) - np.mean(fb_a != ya))
        for m in args.m_grid:
            accepted = violations = 0
            for rep in range(args.reps):
                seed = int(rng.integers(2**62))
                X, y, _ = make_dataset(TASK, m, N_COEFFS, seed=seed)
                d = certify_release(cand(X), fb(X), y,
                                    epsilon=args.epsilon, delta=args.delta)
                if d.released == "candidate":
                    accepted += 1
                    # independent audit: does the RELEASED artifact
                    # violate the population bound?
                    if true_delta > args.epsilon:
                        violations += 1
            rows.append({
                "q": q, "true_delta": true_delta, "m": m,
                "reps": args.reps, "accepted": accepted,
                "violations_among_released": violations,
            })
            print(f"{q:>6.3f}{true_delta:>+12.4f}{m:>8}"
                  f"{accepted / args.reps:>13.1%}"
                  f"{f'{violations}/{accepted}':>21}")
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=30)
    ap.add_argument("--epsilon", type=float, default=0.05)
    ap.add_argument("--delta", type=float, default=0.05)
    ap.add_argument("--m-grid", type=int, nargs="+",
                    default=[2000, 20000], dest="m_grid")
    ap.add_argument("--q-grid", type=float, nargs="+",
                    default=[0.0, 0.02, 0.04, 0.055, 0.07, 0.10, 0.20],
                    dest="q_grid")
    args = ap.parse_args()

    bench = load_benchmark(TASK, n_coeffs=N_COEFFS)
    report: dict = {}

    print("PART 1: adaptive attacker (data_adaptive) end to end")
    g = adaptive_attacker_vs_guard(bench)
    print(f"  guard: {g['poisoned_calls']} poisoned responses, "
          f"{g['guard_violations']} violations, detected="
          f"{g['guard_detected']}   <- 0 detections is the theorem, "
          f"now exhibited by running code")
    learner = credulous_learner_under_attack(bench)
    print(f"  credulous learner: true accuracy "
          f"{learner['true_accuracy']:.4f} "
          f"(in the attacker's world: "
          f"{learner['accuracy_in_attackers_world']:.4f}) "
          f"[ceiling {bench.bayes_accuracy:.4f}, "
          f"1-ceiling {1 - bench.bayes_accuracy:.4f}]")
    gated = gate_the_attacked_learner(
        learner.pop("model"), args.epsilon, args.delta, 2000, 555_001
    )
    print(f"  gate: released={gated['released']}, released audit acc "
          f"{gated['released_audit_accuracy']:.4f}, excess over fallback "
          f"on independent 20k audit "
          f"{gated['released_excess_on_independent_audit']:+.4f}")
    report["part1_adaptive_attack"] = {
        "guard": g, "credulous_learner": learner, "gate": gated,
    }

    report["part2_operating_characteristics"] = operating_characteristics(
        args
    )
    report["settings"] = vars(args)

    path = OUT / "adversary_eval.json"
    path.write_text(json.dumps(report, indent=1))
    print(f"\nresults -> {path}")


if __name__ == "__main__":
    main()
