"""Sequential gullibility: many scored decisions, compounding trust errors.

The closest match we have to the published gullibility experiment. The
agent classifies fields one at a time, consulting four scouts that observe
coefficients of the field currently under consideration, one of which lies
at a fixed rate. Every decision is scored, so a misplaced trust costs a
decision rather than being amortized across a single rule.

Reporting note: accuracy per episode is quantized to 1/n_decisions, so
decisions are POOLED across episodes within a condition and reported with a
binomial confidence interval. Their study runs n=400 episodes per model,
which we cannot match, so pooling is how we recover usable resolution.

Results -> experiments/outputs/sequential_gullibility.json

Run:  python experiments/sequential_gullibility_eval.py
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
from galois_ml.sequential_scouts import run_sequential_episode

OUT = Path("experiments/outputs")
N_COEFFS = 1000
DEFAULT_RATES = [0.0, 0.10, 0.25, 0.50]
DEFAULT_MODELS = [
    "us.anthropic.claude-sonnet-4-6",
    "us.anthropic.claude-opus-4-8",
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
    ap.add_argument("--seeds", type=int, default=3)
    # more decisions per episode buys resolution more cheaply than more
    # seeds, since per-episode accuracy is quantized to 1/n_decisions
    ap.add_argument("--decisions", type=int, default=20)
    # 30, not 8. At 8 the agent cannot observe enough prime powers to
    # determine the group at all: it returned a constant label for every
    # field and scored 0.250, which would have been misread as extreme
    # gullibility when it was observation starvation. The rate-0 control
    # validates the budget: at 30 calls an honest agent reaches 0.875
    # against a 0.9518 ceiling, so the ceiling comparison is meaningful.
    ap.add_argument("--calls-per-decision", type=int, default=30)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument(
        "--out", default=None,
        help="results path; set per shard so parallel workers do not "
             "overwrite one another",
    )
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    out = Path(args.out) if args.out else OUT / "sequential_gullibility.json"
    results = []
    bench = load_benchmark(args.task, n_coeffs=N_COEFFS)
    episodes = (len(args.models) * len(args.rates) * len(args.roles)
                * args.seeds)
    print(f"{episodes} episodes x {args.decisions} decisions = "
          f"{episodes * args.decisions} scored decisions")
    print(f"task {args.task} (ceiling {bench.bayes_accuracy:.4f}), "
          f"liar {args.liar}, rates {args.rates}\n")

    for model_id in args.models:
        label = model_id.replace("us.anthropic.claude-", "")
        print(f"{'=' * 66}\n== {label}\n{'=' * 66}")
        for role in args.roles:
            for rate in args.rates:
                for seed in range(args.seeds):
                    res = run_sequential_episode(
                        bench, model_id, lie_rate=rate, liar=args.liar,
                        role=role, region=args.region, seed=seed,
                        temperature=args.temperature,
                        n_decisions=args.decisions,
                        calls_per_decision=args.calls_per_decision,
                    )
                    results.append(res)
                    out.write_text(
                        json.dumps([r.to_dict() for r in results], indent=1)
                    )
                    if res.valid:
                        attr = ("CORRECT" if res.correct_attribution
                                else f"accused={res.accused}")
                        print(f"  {role:<10} lie {rate:<5} s{seed} "
                              f"{res.n_correct}/{res.n_decisions} "
                              f"= {res.accuracy:.3f}  lies {res.lies_told:<3} "
                              f"{attr}")
                    else:
                        print(f"  {role:<10} lie {rate:<5} s{seed} "
                              f"FAILED [{res.error}]")

    print(f"\nresults -> {out}")


if __name__ == "__main__":
    main()
