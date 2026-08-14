"""Gullibility curve with provable ground truth.

Directly comparable to the gullibility curve in Anthropic's Frontier Red
Team report (2026-08-13), which plots routing-decision accuracy against
how often an untrustworthy scout lies, bracketed by two baselines:
"trust everyone" (believe every report despite contradictions) and
"learns who lies" (discard the liar once identifiable).

The analogue here:

  x-axis   poison_rate: the probability that any given tool call returns
           falsified data rather than the truth
  lower    role=credulous: an agent instructed to treat tool output as
           authoritative. Measured, not modelled.
  upper    poison_rate=0 accuracy: the clean-tool control, i.e. what an
           agent achieves if it perfectly discounts corrupted evidence
  curves   role=analyst: the ordinary agent, per model generation

What our version adds: the upper baseline is not merely a
better-behaved policy, it is the provable Bayes-optimal ceiling for the
task, computed in closed form. So the vertical axis is calibrated
absolutely rather than relative to a hand-built scenario.

The `data` channel is used because "believing the lie" has a well-defined
consequence there: the falsified distributions imply the label-inverted
rule, so credulity is punished in a measurable way.

Results -> experiments/outputs/gullibility.json

Run:  python experiments/gullibility_eval.py [--models ...] [--seeds 3]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Make the repo importable regardless of interpreter or install state.
import pathlib as _pathlib
sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parent.parent))

from galois_ml._deps import require_agents
from galois_ml.benchmark import load_benchmark
from galois_ml.tool_poisoning import run_poisoned_pipeline

OUT = Path("experiments/outputs")
N_COEFFS = 1000

#: Matches the lie rates on Anthropic's x-axis.
DEFAULT_RATES = [0.0, 0.10, 0.25, 0.50]

#: Generations overlapping their figure, oldest to newest.
DEFAULT_MODELS = [
    "us.anthropic.claude-sonnet-4-6",
    "us.anthropic.claude-sonnet-5",
    "us.anthropic.claude-opus-4-6-v1",
    "us.anthropic.claude-opus-4-8",
]


def main() -> None:
    require_agents()
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    ap.add_argument("--region", default="us-east-1")
    ap.add_argument("--task", default="sextic")
    ap.add_argument("--attack", default="data")
    ap.add_argument("--rates", nargs="+", type=float, default=DEFAULT_RATES)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--roles", nargs="+", default=["analyst", "credulous"])
    ap.add_argument("--max-tool-calls", type=int, default=None)
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / "gullibility.json"
    results = []
    bench = load_benchmark(args.task, n_coeffs=N_COEFFS)

    total = (len(args.models) * len(args.rates) * len(args.roles)
             * args.seeds)
    print(f"{total} runs on {args.task} (ceiling {bench.bayes_accuracy:.4f}), "
          f"attack={args.attack}")
    print(f"rates {args.rates}, roles {args.roles}, seeds {args.seeds}")

    for model_id in args.models:
        label = model_id.replace("us.anthropic.claude-", "").replace("-v1:0", "")
        print(f"\n{'=' * 68}\n== {label}\n{'=' * 68}")
        for role in args.roles:
            for rate in args.rates:
                for seed in range(args.seeds):
                    res = run_poisoned_pipeline(
                        bench, model_id,
                        # rate 0 is the clean control regardless of channel
                        attack="none" if rate == 0.0 else args.attack,
                        with_critic=False,
                        region=args.region, seed=seed,
                        theory_in_brief=True,
                        temperature=args.temperature,
                        max_tool_calls=args.max_tool_calls,
                        poison_rate=rate,
                        role=role,
                    )
                    results.append(res)
                    out.write_text(
                        json.dumps([r.to_dict() for r in results], indent=1)
                    )
                    if res.valid:
                        flag = "FLAGGED" if res.attack_surfaced else "-"
                        print(f"  {role:<10} rate {rate:<5} s{seed} "
                              f"acc {res.test_accuracy:.4f} "
                              f"regret {res.regret:+.4f} {flag}")
                    else:
                        print(f"  {role:<10} rate {rate:<5} s{seed} "
                              f"FAILED [{res.error}]")

    print(f"\nresults -> {out}")
    print("plot with: python experiments/plot_gullibility.py")


if __name__ == "__main__":
    main()
