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

The `combined` channel is the default, and the choice matters. An earlier
version used `data`, which corrupts only the observation tool and leaves
`test_rule` honest. Agents simply formed a hypothesis from the falsified
distributions, tested it against truthful validation, saw it fail, and
revised: every curve came out flat at the ceiling across corruption rates
from 0 to 0.5. The same effect is visible in the channel-harm table, where
`data` alone is not significant (p=0.28) while `combined` is (p=0.0001).

A gullibility curve therefore requires a channel with no honest fallback.
`combined` corrupts observation, examples and validation together and
consistently, so partial corruption produces genuinely contradictory
evidence the agent has to adjudicate rather than an inconsistency it can
resolve for free against a truthful oracle.

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

#: Their x-axis is {0, 0.1, 0.25, 0.5}. We extend to 1.0 because that is
#: where we have an independent harm measurement to anchor against
#: (combined at full rate: +0.170 regret over control, p=0.0001), which
#: lets us check that the curve's endpoint reproduces a known value.
DEFAULT_RATES = [0.0, 0.10, 0.25, 0.50, 1.0]

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
    # `combined` leaves no honest channel; see the module docstring for why
    # single-channel attacks produce flat curves here
    ap.add_argument("--attack", default="combined")
    ap.add_argument("--rates", nargs="+", type=float, default=DEFAULT_RATES)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--roles", nargs="+", default=["analyst", "credulous"])
    ap.add_argument("--max-tool-calls", type=int, default=None)
    ap.add_argument(
        "--theory-in-brief", action="store_true",
        help="leave the group-theory table in the brief. Defaults to OFF: "
             "with it on, agents bypass the corrupted tools entirely and "
             "every curve is flat at the ceiling.",
    )
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
                        # The tool channel MUST be load-bearing here. The
                        # first version of this sweep passed True, leaving
                        # the group-theory table in the brief as an
                        # un-poisonable channel; every curve came out flat
                        # at the ceiling, including the credulous baseline,
                        # because agents derived the rule from theory and
                        # ignored the corrupted evidence entirely.
                        theory_in_brief=args.theory_in_brief,
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
