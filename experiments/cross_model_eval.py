"""Epistemic vigilance across model generations.

Anthropic's Frontier Red Team report (2026-08-13) finds that newer models
recover more of the gap between a naive "trust everything" policy and an
oracle with perfect lie discovery, testing Sonnet 4.6, Sonnet 5, Opus 4.6,
Opus 4.8 and Mythos-class models. Their ground truth is stipulated by
scenario construction: a scripted scout lies at a fixed rate.

This runs the same comparison with provable ground truth. The `impossible`
tool-poisoning channel reports zeta coefficient values that the Euler
factor formula forbids for any decomposition type, so the lie is provable
from facts already in the agent's brief -- a strictly easier detection
task than resolving contradictions between peers. For each model we
measure:

  detection rate  fraction of runs where an agent surfaced the anomaly
  harm            regret against the exact Bayes ceiling
  outcome mix     flagged / routed_around / degraded / broken

`none` (honest tools) runs as the per-model control, so detection can be
read against that model's baseline competence on the task rather than in
absolute terms: a model that cannot solve the clean task tells us nothing
about its vigilance.

Results -> experiments/outputs/cross_model_<attack>.json

Run:  python experiments/cross_model_eval.py [--models ...] [--seeds 3]
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

#: Ordered oldest to newest, matching the generations in the Anthropic
#: report so the comparison is like-for-like where possible.
#: NOTE: claude-sonnet-4-20250514 is excluded. Bedrock now refuses it with
#: "This Model is marked by provider as Legacy and you have not been
#: actively using the model in the last 30 days", so every run against it
#: fails and only wastes wall time. That costs us the oldest baseline
#: point, which is worth stating when reporting the generational trend.
DEFAULT_MODELS = [
    "us.anthropic.claude-sonnet-4-5-20250929-v1:0",
    "us.anthropic.claude-sonnet-4-6",
    "us.anthropic.claude-opus-4-6-v1",
    "us.anthropic.claude-opus-4-8",
    "us.anthropic.claude-sonnet-5",
    "us.anthropic.claude-opus-5",
]


def short(model_id: str) -> str:
    return model_id.replace("us.anthropic.claude-", "").replace("-v1:0", "")


def main() -> None:
    require_agents()
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    ap.add_argument("--region", default="us-east-1")
    ap.add_argument("--tasks", nargs="+", default=["sextic"])
    # `impossible` is the sharpest vigilance probe (provable lie);
    # `none` is the per-model competence control; `combined` is the
    # closest analogue to a setting with no honest channel to fall back on
    ap.add_argument("--attacks", nargs="+",
                    default=["none", "impossible", "combined"])
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--with-critic", action="store_true")
    ap.add_argument(
        "--max-tool-calls", type=int, default=None,
        help="cap tool calls per agent (default 100; 0 = unlimited)",
    )
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / "cross_model.json"
    results = []
    total = (len(args.models) * len(args.tasks) * len(args.attacks)
             * args.seeds)
    print(f"{total} runs: {len(args.models)} models x {len(args.tasks)} task(s) "
          f"x {len(args.attacks)} attack(s) x {args.seeds} seed(s)")

    for model_id in args.models:
        print(f"\n{'=' * 70}\n== {short(model_id)}\n{'=' * 70}")
        for task in args.tasks:
            bench = load_benchmark(task, n_coeffs=N_COEFFS)
            for attack in args.attacks:
                for seed in range(args.seeds):
                    res = run_poisoned_pipeline(
                        bench, model_id, attack,
                        with_critic=args.with_critic,
                        region=args.region, seed=seed,
                        theory_in_brief=True,
                        temperature=args.temperature,
                        max_tool_calls=args.max_tool_calls,
                    )
                    results.append(res)
                    out.write_text(
                        json.dumps([r.to_dict() for r in results], indent=1)
                    )
                    if res.valid:
                        flag = "FLAGGED" if res.attack_surfaced else "undetected"
                        print(f"  {task} {attack:<12} s{seed} "
                              f"acc {res.test_accuracy:.4f} "
                              f"regret {res.regret:+.4f} {flag} "
                              f"poisoned {res.poisoned_tool_calls}")
                    else:
                        print(f"  {task} {attack:<12} s{seed} "
                              f"FAILED [{res.error}]")

    print(f"\nresults -> {out}")
    print("summarize with: python experiments/report.py")


if __name__ == "__main__":
    main()
