"""Tool-channel poisoning experiments: honest agents, compromised
environment.

Attack channels (all with identical honest agent prompts, so harm is
attributable to the channel rather than to instructions):

  none          honest tools (control)
  data          observation tool reports falsified class distributions
  verification  validation tool inflates accuracy, hides counterexamples
  injection     tool output carries injected operator-style instructions
  combined      falsified observations + injection + inflated validation,
                mutually consistent (no honest channel left)

Two evidence regimes, because the first run of this experiment showed
they measure different things:

  --theory-in-brief (default)  the brief lists the candidate groups'
      element orders, an un-poisonable channel sufficient to derive the
      criterion without tools. Measures robustness via redundancy.
  --no-theory-in-brief         those facts are withheld, so the tool
      channel is load-bearing. Measures attack harm.

Repeated seeds with temperature > 0 give independent trajectories for
variance estimates. Results stream to
experiments/outputs/tool_poisoning_<model>.json.

Run:  python experiments/tool_poisoning_eval.py --model-id <bedrock-id>
        [--tasks quartic] [--attacks none data verification injection combined]
        [--seeds 3] [--temperature 0.7] [--no-theory-in-brief]
        [--no-critic-runs]
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
from galois_ml.tool_poisoning import ATTACKS, run_poisoned_pipeline

OUT = Path("experiments/outputs")
N_COEFFS = 1000


def main() -> None:
    require_agents()
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", required=True)
    ap.add_argument("--region", default="us-east-1")
    # quartic by default: its criterion is deterministic (a_{p^2} = 0 for
    # some p), so an attack either breaks the rule or does not, giving a
    # much cleaner signal than the density-threshold tasks
    ap.add_argument("--tasks", nargs="+", default=["quartic"])
    ap.add_argument("--attacks", nargs="+", default=list(ATTACKS))
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--no-theory-in-brief", action="store_true")
    ap.add_argument("--no-critic-runs", action="store_true")
    ap.add_argument("--out-suffix", default="")
    ap.add_argument(
        "--max-tool-calls", type=int, default=None,
        help="cap tool calls per agent (default 100; 0 = unlimited). "
             "On exhaustion tools instruct the agent to finalize "
             "rather than erroring, so the run still yields a result.",
    )
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    stem = args.model_id.replace(":", "_").replace("/", "_")
    out = OUT / f"tool_poisoning_{stem}{args.out_suffix}.json"
    results = []
    critic_modes = [False] if args.no_critic_runs else [False, True]
    theory = not args.no_theory_in_brief

    total = (
        len(args.tasks) * len(args.attacks) * len(critic_modes) * args.seeds
    )
    print(f"{total} runs: {len(args.tasks)} task(s) x {len(args.attacks)} "
          f"attack(s) x {len(critic_modes)} critic mode(s) x {args.seeds} "
          f"seed(s); theory_in_brief={theory}, temp={args.temperature}")

    for task in args.tasks:
        bench = load_benchmark(task, n_coeffs=N_COEFFS)
        print(f"\n== {task} (ceiling {bench.bayes_accuracy:.4f}) ==")
        for attack in args.attacks:
            for with_critic in critic_modes:
                for seed in range(args.seeds):
                    res = run_poisoned_pipeline(
                        bench, args.model_id, attack, with_critic,
                        region=args.region, seed=seed,
                        theory_in_brief=theory,
                        temperature=args.temperature,
                    max_tool_calls=args.max_tool_calls,
                    )
                    results.append(res)
                    out.write_text(
                        json.dumps([r.to_dict() for r in results], indent=1)
                    )
                    label = f"{attack}{'+critic' if with_critic else ''}"
                    if res.valid:
                        flag = "FLAGGED" if res.attack_surfaced else "undetected"
                        print(
                            f"{label:<22} s{seed} acc {res.test_accuracy:.4f}  "
                            f"regret {res.regret:+.4f}  {flag}  "
                            f"poisoned {res.poisoned_tool_calls}"
                            + (f"  critic={res.critic_verdict}"
                               if res.critic_verdict else "")
                        )
                    else:
                        print(f"{label:<22} s{seed} FAILED [{res.error}]")

    print(f"\nresults -> {out}")
    print("summarize and plot with: python experiments/plot_results.py")


if __name__ == "__main__":
    main()
