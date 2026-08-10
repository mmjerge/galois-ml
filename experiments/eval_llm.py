"""Run the LLM evaluation harness on the Galois benchmarks.

Without --model-id, runs the offline controls (theorem oracle and
majority floor) to validate the harness end to end. With --model-id, it
evaluates that Bedrock model on every task under the rule_induction and
derivation protocols (held-out tasks degree14/21/25 are the
contamination-resistant ones).

Results -> experiments/outputs/llm_eval_<model>.json

Examples:
  python experiments/eval_llm.py                                # controls
  python experiments/eval_llm.py --model-id us.anthropic.claude-sonnet-4-20250514-v1:0
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from galois_ml.benchmark import load_benchmark
from galois_ml.groups import PAPER_TASKS, TASKS
from galois_ml.llm_eval import (
    BedrockModel,
    MajorityModel,
    OracleModel,
    _ORACLE_EXPONENT,
    evaluate_model,
)

OUT = Path("experiments/outputs")
N_COEFFS = 1000


def _print(res) -> None:
    acc = "  FAIL " if not res.valid_response else f"{res.accuracy:.4f}"
    regret = " " * 7 if res.regret is None else f"{res.regret:+.4f}"
    held = " " if res.task in PAPER_TASKS else "*"
    print(f"{res.task:<18}{held} {res.protocol:<22} {res.model_name:<12} "
          f"acc {acc}  regret {regret}"
          + (f"  [{res.error}]" if res.error else ""))


def run_controls() -> list:
    results = []
    for task in _ORACLE_EXPONENT:  # tasks with a wired oracle rule
        bench = load_benchmark(task, n_coeffs=N_COEFFS)
        for model_name, model in [
            (f"oracle", OracleModel(task)),
            ("majority", MajorityModel()),
        ]:
            res = evaluate_model(
                model, bench, "rule_induction", model_name=model_name
            )
            results.append(res)
            _print(res)
    return results


def run_bedrock(model_id: str, region: str) -> list:
    model = BedrockModel(model_id, region=region)
    results = []
    for task in TASKS:
        bench = load_benchmark(task, n_coeffs=N_COEFFS)
        for protocol in ("rule_induction", "derivation"):
            res = evaluate_model(
                model, bench, protocol, model_name=model_id.split(".")[-1]
            )
            results.append(res)
            _print(res)
    return results


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", default=None)
    ap.add_argument("--region", default="us-east-1")
    args = ap.parse_args()

    print("(* = held-out, contamination-resistant task)\n")
    if args.model_id:
        results = run_bedrock(args.model_id, args.region)
        stem = args.model_id.replace(":", "_").replace("/", "_")
    else:
        results = run_controls()
        stem = "controls"

    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / f"llm_eval_{stem}.json"
    out.write_text(json.dumps([r.to_dict() for r in results], indent=1))
    print(f"\nresults -> {out}")


if __name__ == "__main__":
    main()
