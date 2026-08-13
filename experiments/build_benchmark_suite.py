"""Build the full Galois interpretability benchmark suite for
Hugging Face: parquet data files + JSON metadata per config, plus a
top-level dataset card and manifest.

Configs: all 6 Galois group tasks x feature budgets N in {100, 1000}.
The feature budget controls the Bayes ceiling (fewer visible prime
powers -> harder task), and the ceiling for every config is recorded to
4-decimal precision in metadata.json.

Run:  python experiments/build_benchmark_suite.py [out_dir]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Make the repo importable regardless of interpreter or install state.
import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parent.parent))

from galois_ml.benchmark import export_benchmark, load_benchmark
from galois_ml.groups import PAPER_TASKS, TASKS

FEATURE_BUDGETS = [100, 1000]
N_TRAIN, N_TEST = 2400, 600
DEFAULT_OUT = "benchmarks/galois-groups-v1"

CARD = """\
# Galois Group Interpretability Benchmark (v1)

Classify the Galois group of a (virtual) Galois number field K/Q from
its Dedekind zeta coefficients a_1..a_N. Data follows the exact
generative model of unramified prime decomposition (Chebotarev), after
Lee & Lee, "Machines Learn Number Fields, But How?" (arXiv:2508.06670).

What makes this a benchmark rather than a dataset: every config ships
provable ground truth in metadata.json -

* `bayes_accuracy`: the Bayes-optimal accuracy (ceiling for ANY model),
* `feature_mutual_information`: exact I(a_n; label) for every feature,
  the true attribution ranking,
* `explanation`: the number-theoretic theorem a faithful model should
  rediscover.

Intended uses: evaluating interpretability/attribution methods against
exact ground truth; measuring accuracy regret against a known ceiling;
sanity-checking symmetry- and structure-discovery methods.

Each config directory contains train.parquet / test.parquet with
columns `zeta` (list<float64>), `label` (int64), `group` (string).
"""


def main() -> None:
    out_root = Path(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_OUT)
    manifest: list[dict] = []
    for task in TASKS:
        for n_coeffs in FEATURE_BUDGETS:
            name = f"{task}_n{n_coeffs}"
            print(f"building {name} ...", flush=True)
            bench = load_benchmark(
                task, n_train=N_TRAIN, n_test=N_TEST, n_coeffs=n_coeffs
            )
            export_benchmark(bench, out_root / name)
            manifest.append({
                "config": name,
                "task": task,
                "classes": bench.class_names,
                "n_coeffs": n_coeffs,
                "bayes_accuracy": round(bench.bayes_accuracy, 4),
                "bayes_accuracy_se": round(bench.bayes_accuracy_se, 5),
                # held-out tasks are absent from arXiv:2508.06670 and
                # any other publication: contamination-resistant.
                "held_out": task not in PAPER_TASKS,
            })
            print(f"  ceiling {bench.bayes_accuracy:.4f} "
                  f"(+/-{bench.bayes_accuracy_se:.4f})")

    (out_root / "manifest.json").write_text(json.dumps(manifest, indent=1))
    (out_root / "README.md").write_text(CARD)
    print(f"\nsuite written to {out_root}/ "
          f"({len(manifest)} configs + manifest.json + README.md)")


if __name__ == "__main__":
    main()
