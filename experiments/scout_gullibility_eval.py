"""Gullibility with an identifiable unreliable source.

Structural analogue of the gullibility experiment in Anthropic's Frontier
Red Team report: an agent decides from four peer sources with overlapping
coverage, one of which lies at a fixed rate, and a lie is detectable
because a peer covering the same index contradicts it.

This supersedes experiments/gullibility_eval.py, which varied a per-call
corruption probability across all tools. That is environment noise, not a
lying source: there is no persistent identity to distrust, so "which source
is unreliable" cannot be asked, and the incoherent per-call flicker
produced a non-monotonic baseline that no persistent liar can produce.

Two measurements per episode:

  accuracy      task accuracy against the exact Bayes ceiling, as a
                function of the liar's lie rate (their y-axis)
  attribution   whether the agent NAMES the unreliable scout, which is
                asked directly rather than inferred from keywords. This is
                the sharper question: an agent can be unharmed while having
                no idea which source was bad.

Baselines bracketing the curve, as in their figure:
  credulous     an agent told every scout is authoritative (their "trust
                everyone")
  ceiling       the provable Bayes optimum (their "learns who lies", except
                ours is a computed bound rather than a policy)

Results -> experiments/outputs/scout_gullibility.json

Run:  python experiments/scout_gullibility_eval.py [--models ...] [--seeds 4]
"""

from __future__ import annotations

import argparse
import json
import pathlib as _pathlib
import sys
from pathlib import Path

sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parent.parent))

from galois_ml._deps import require_agents
from galois_ml.benchmark import load_benchmark
from galois_ml.scouts import run_scout_episode

OUT = Path("experiments/outputs")
N_COEFFS = 1000
DEFAULT_RATES = [0.0, 0.10, 0.25, 0.50]
DEFAULT_MODELS = [
    "us.anthropic.claude-sonnet-4-6",
    "us.anthropic.claude-opus-4-8",
    "us.anthropic.claude-opus-5",
]


def main() -> None:
    require_agents()
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    ap.add_argument("--region", default="us-east-1")
    ap.add_argument("--task", default="sextic")
    ap.add_argument("--rates", nargs="+", type=float, default=DEFAULT_RATES)
    ap.add_argument("--roles", nargs="+", default=["listener", "credulous"])
    ap.add_argument("--liar", default="scout_c")
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--max-tool-calls", type=int, default=60)
    ap.add_argument(
        "--one-query-per-index", action="store_true",
        help="answer only the first scout asked about each coefficient. "
             "Closes a consistency gap with the published setup, whose "
             "listener cannot re-ask: without it an agent can poll every "
             "covering scout and average the disagreement away.",
    )
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    suffix = "_budgeted" if args.one_query_per_index else ""
    out = OUT / f"scout_gullibility{suffix}.json"
    results = []
    bench = load_benchmark(args.task, n_coeffs=N_COEFFS)
    total = (len(args.models) * len(args.rates) * len(args.roles)
             * args.seeds)
    print(f"{total} episodes on {args.task} "
          f"(ceiling {bench.bayes_accuracy:.4f}), liar={args.liar}")
    print(f"rates {args.rates}, roles {args.roles}, seeds {args.seeds}\n")

    for model_id in args.models:
        label = model_id.replace("us.anthropic.claude-", "")
        print(f"{'=' * 66}\n== {label}\n{'=' * 66}")
        for role in args.roles:
            for rate in args.rates:
                for seed in range(args.seeds):
                    res = run_scout_episode(
                        bench, model_id, lie_rate=rate, liar=args.liar,
                        role=role, region=args.region, seed=seed,
                        temperature=args.temperature,
                        max_tool_calls=args.max_tool_calls,
                        one_query_per_index=args.one_query_per_index,
                    )
                    results.append(res)
                    out.write_text(
                        json.dumps([r.to_dict() for r in results], indent=1)
                    )
                    if res.valid:
                        attr = ("CORRECT" if res.correct_attribution
                                else f"accused={res.accused}")
                        print(f"  {role:<10} lie {rate:<5} s{seed} "
                              f"acc {res.test_accuracy:.4f} "
                              f"lies_told {res.lies_told:<3} {attr}")
                    else:
                        print(f"  {role:<10} lie {rate:<5} s{seed} "
                              f"FAILED [{res.error}]")

    print(f"\nresults -> {out}")


if __name__ == "__main__":
    main()
