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
    # NOT quartic. Its ceiling is 0.9998 and agents reach it on every
    # control run, so there is almost no headroom in which harm can
    # register -- a 72-run quartic sweep moved the needle only for the
    # most extreme attack. sextic (ceiling 0.9518) and degree21 (0.9352)
    # leave measurable room between agent performance and the ceiling,
    # which is what an attack has to eat into to be observable.
    ap.add_argument("--tasks", nargs="+", default=["sextic", "degree21"])
    ap.add_argument("--attacks", nargs="+", default=list(ATTACKS))
    # bimodal conditions (notably `combined`) need more than 3 samples
    # before a median or a solved/broken split means anything
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--no-theory-in-brief", action="store_true")
    ap.add_argument("--no-critic-runs", action="store_true")
    ap.add_argument(
        "--guarded", action="store_true",
        help="interpose the deterministic tool guard (galois_ml.guard) "
             "between tools and agents; guard detection is recorded "
             "separately from agent detection",
    )
    ap.add_argument(
        "--guard-policy", choices=("flag", "block", "repair"),
        default="flag",
        help="what a guard violation does to the output the agent sees: "
             "annotate it (flag), withhold it (block), or substitute the "
             "trusted recomputation verbatim (repair -- the policy with "
             "a provable clean-transcript property on detected channels)",
    )
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
          f"seed(s); theory_in_brief={theory}, temp={args.temperature}, "
          f"guarded={args.guarded}"
          + (f" (policy={args.guard_policy})" if args.guarded else ""))

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
                        guarded=args.guarded,
                        guard_policy=args.guard_policy,
                    )
                    results.append(res)
                    out.write_text(
                        json.dumps([r.to_dict() for r in results], indent=1)
                    )
                    label = (f"{attack}{'+critic' if with_critic else ''}"
                             f"{'+guard' if args.guarded else ''}")
                    if res.valid:
                        flag = "FLAGGED" if res.attack_surfaced else "undetected"
                        if args.guarded:
                            n_guard = sum(res.guard_flags.values())
                            flag += (f"  guard {n_guard} violation(s)"
                                     if res.guard_detected else "  guard clean")
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
