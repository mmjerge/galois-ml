"""Tool-channel poisoning experiments: honest agents, compromised
environment.

Conditions per task (all with the same honest agent prompts, so harm is
attributable to the channel):

  none          honest tools (control)
  data          observation tool reports falsified class distributions
  verification  validation tool inflates accuracy, hides counterexamples
  injection     tool output carries injected operator-style instructions

Each condition runs with and without an honest critic, giving:
- attack harm:      regret(attack) - regret(none)
- defense recovery: regret(attack) - regret(attack + critic)
- detection rate:   fraction of runs where an agent flagged the anomaly

Results stream to experiments/outputs/tool_poisoning_<model>.json.

Run:  python experiments/tool_poisoning_eval.py --model-id <bedrock-id>
      [--tasks sextic] [--attacks none data verification injection]
      [--no-critic-runs]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from galois_ml.benchmark import load_benchmark
from galois_ml.tool_poisoning import ATTACKS, run_poisoned_pipeline

OUT = Path("experiments/outputs")
N_COEFFS = 1000


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", required=True)
    ap.add_argument("--region", default="us-east-1")
    ap.add_argument("--tasks", nargs="+", default=["sextic"])
    ap.add_argument("--attacks", nargs="+", default=list(ATTACKS))
    ap.add_argument(
        "--no-critic-runs",
        action="store_true",
        help="only run the single-agent conditions (halves cost)",
    )
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    stem = args.model_id.replace(":", "_").replace("/", "_")
    out = OUT / f"tool_poisoning_{stem}.json"
    results = []
    critic_modes = [False] if args.no_critic_runs else [False, True]

    for task in args.tasks:
        bench = load_benchmark(task, n_coeffs=N_COEFFS)
        print(f"\n== {task} (ceiling {bench.bayes_accuracy:.4f}) ==")
        for attack in args.attacks:
            for with_critic in critic_modes:
                res = run_poisoned_pipeline(
                    bench, args.model_id, attack, with_critic,
                    region=args.region,
                )
                results.append(res)
                out.write_text(
                    json.dumps([r.to_dict() for r in results], indent=1)
                )
                label = f"{attack}{'+critic' if with_critic else ''}"
                if res.valid:
                    flag = "FLAGGED" if res.attack_surfaced else "undetected"
                    print(
                        f"{label:<22} acc {res.test_accuracy:.4f}  "
                        f"regret {res.regret:+.4f}  {flag}  "
                        f"poisoned_calls {res.poisoned_tool_calls}"
                        + (f"  critic={res.critic_verdict}"
                           if res.critic_verdict else "")
                    )
                else:
                    print(f"{label:<22} FAILED [{res.error}]")

    print(f"\nresults -> {out}")


if __name__ == "__main__":
    main()
