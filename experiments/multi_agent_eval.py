"""Multi-agent discovery experiments on the Galois benchmarks (Strands
agents on Bedrock, with data-analysis tools).

Four conditions per task, chosen to make trust properties measurable:

  analyst          tool-using single agent (baseline)
  analyst_critic   proposer + independent verifier
  saboteur         simulated compromised analyst, unreviewed
  saboteur_critic  compromised analyst + verifier (detection test)

Ground truth turns the comparisons into exact measurements:
- (analyst_critic - analyst): the value added by verification
- (saboteur - analyst): harm caused by a compromised member
- (saboteur_critic - saboteur): harm recovered by verification

Results stream to experiments/outputs/multi_agent_<model>.json.

Run:  python experiments/multi_agent_eval.py --model-id <bedrock-id>
      [--tasks sextic decic degree21] [--conditions ...]
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
from galois_ml.multi_agent import CONDITIONS, run_pipeline

OUT = Path("experiments/outputs")
DEFAULT_TASKS = ["sextic", "decic", "degree21"]
N_COEFFS = 1000


def main() -> None:
    require_agents()
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", required=True)
    ap.add_argument("--region", default="us-east-1")
    ap.add_argument("--tasks", nargs="+", default=DEFAULT_TASKS)
    ap.add_argument("--conditions", nargs="+", default=list(CONDITIONS))
    ap.add_argument("--seeds", type=int, default=5)
    # temperature > 0 is required for repeated seeds to be informative:
    # at 0 the runs would be near-identical and the spread meaningless
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument(
        "--max-tool-calls", type=int, default=None,
        help="cap tool calls per agent (default 100; 0 = unlimited). "
             "On exhaustion tools instruct the agent to finalize "
             "rather than erroring, so the run still yields a result.",
    )
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    stem = args.model_id.replace(":", "_").replace("/", "_")
    out = OUT / f"multi_agent_{stem}.json"
    results = []

    total = len(args.tasks) * len(args.conditions) * args.seeds
    print(f"{total} runs: {len(args.tasks)} task(s) x {len(args.conditions)} "
          f"condition(s) x {args.seeds} seed(s), temp {args.temperature}")

    for task in args.tasks:
        bench = load_benchmark(task, n_coeffs=N_COEFFS)
        print(f"\n== {task} (ceiling {bench.bayes_accuracy:.4f}) ==")
        for condition in args.conditions:
            for seed in range(args.seeds):
                res = run_pipeline(
                    bench, args.model_id, condition,
                    region=args.region,
                    max_tool_calls=args.max_tool_calls,
                    seed=seed,
                    temperature=args.temperature,
                )
                results.append(res)
                out.write_text(
                    json.dumps([r.to_dict() for r in results], indent=1)
                )
                if res.valid:
                    verdict = (f" critic={res.critic_verdict}"
                               if res.critic_verdict else "")
                    print(f"{condition:<16} s{seed} acc {res.test_accuracy:.4f}  "
                          f"regret {res.regret:+.4f}{verdict}  "
                          f"tools {res.tool_calls}")
                else:
                    print(f"{condition:<16} s{seed} FAILED [{res.error}]")

    print(f"\nresults -> {out}")


if __name__ == "__main__":
    main()
