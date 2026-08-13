"""Does agent redundancy buy independence?

Runs n independent analyst agents on the same task and measures whether
their failures are correlated. This tests the premise underlying every
multi-agent assurance argument (voting, review, Byzantine tolerance):
that independent agents fail independently.

Motivated by Anthropic's Frontier Red Team findings (Aug 2026) that
agents are "low variance" and tend to make the same bad decision, so
that individually benign quirks compound into systemic failure. Here the
effect is quantified against an exact ceiling:

  correlation_ratio            observed error overlap / overlap if independent
                               (1.0 = independent, >1 = correlated failure)
  effective_independent_agents how many truly independent agents the
                               deployed ensemble was worth
  realized_gain_fraction       fraction of the theoretical ensembling
                               benefit actually obtained

Results -> experiments/outputs/conformity_<model>.json

Run:  python experiments/conformity_eval.py --model-id <bedrock-id>
        [--tasks quartic sextic] [--agents 5] [--temperature 1.0]
        [--no-theory-in-brief]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

# Make the repo importable regardless of interpreter or install state.
import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parent.parent))

from galois_ml._deps import require_agents
from galois_ml.benchmark import load_benchmark
from galois_ml.conformity import run_conformity_experiment

OUT = Path("experiments/outputs")
N_COEFFS = 1000


def main() -> None:
    require_agents()
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", required=True)
    ap.add_argument("--region", default="us-east-1")
    ap.add_argument("--tasks", nargs="+", default=["sextic"])
    ap.add_argument("--agents", type=int, default=5)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--no-theory-in-brief", action="store_true")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    stem = args.model_id.replace(":", "_").replace("/", "_")
    out = OUT / f"conformity_{stem}.json"
    results = []

    for task in args.tasks:
        bench = load_benchmark(task, n_coeffs=N_COEFFS)
        print(f"\n== {task}: {args.agents} independent agents, "
              f"temp {args.temperature}, ceiling {bench.bayes_accuracy:.4f} ==")
        res = run_conformity_experiment(
            bench, args.model_id, n_agents=args.agents, region=args.region,
            temperature=args.temperature,
            theory_in_brief=not args.no_theory_in_brief,
        )
        results.append(res)
        out.write_text(json.dumps([r.to_dict() for r in results], indent=1))

        for r in res.runs:
            status = (f"acc {r['accuracy']:.4f}" if r["valid"]
                      else f"FAILED [{r['error']}]")
            print(f"  agent {r['seed']}: {status}")
        m = res.metrics
        if "note" in m:
            print(f"  {m['note']}")
            continue
        print(f"\n  individual accuracy: {m['individual_accuracy_mean']:.4f} "
              f"+/- {m['individual_accuracy_std']:.4f} "
              f"(min {m['individual_accuracy_min']:.4f}, "
              f"max {m['individual_accuracy_max']:.4f})")
        print(f"  distinct conjectures: {m['distinct_conjectures']} "
              f"of {m['n_valid_agents']}")
        print(f"  error overlap (Jaccard): {m['observed_jaccard']:.3f} "
              f"observed vs {m['independent_jaccard']:.3f} if independent"
              f"  -> correlation ratio {m['correlation_ratio']:.2f}x")
        print(f"  majority vote: {m['majority_vote_accuracy']:.4f} "
              f"(independent ensemble would give "
              f"{m['independent_ensemble_accuracy']:.4f})")
        rg = m["realized_gain_fraction"]
        print(f"  realized ensembling gain: "
              + (f"{rg:.1%} of theoretical" if rg is not None else "n/a"))
        print(f"  effective independent agents: "
              f"{m['effective_independent_agents']:.2f} "
              f"(deployed {m['n_valid_agents']})")

    print(f"\nresults -> {out}")


if __name__ == "__main__":
    main()
