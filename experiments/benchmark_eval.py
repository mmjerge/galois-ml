"""Score standard models against the Galois benchmarks' provable ground
truth: accuracy regret vs the Bayes ceiling, and feature-attribution
quality vs the exact mutual-information ranking.

This is the benchmark use case: because the generative model is known,
we can measure not just whether a model is accurate but whether its
*explanation* (feature importances) points at the genuinely informative
features.

Writes results to experiments/outputs/benchmark_eval.json.

Run:  python experiments/benchmark_eval.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

# Make the repo importable regardless of interpreter or install state.
import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parent.parent))

from galois_ml.benchmark import load_benchmark
from galois_ml.groups import TASKS

TOP_K = 20
OUT = Path("experiments/outputs")


def main() -> None:
    results: list[dict] = []
    header = (
        f"{'task':<18} {'model':<8} {'acc':>7} {'regret':>8} "
        f"{'spearman':>9} {'prec@20':>8} {'MIrec@20':>9}"
    )
    print(header)
    print("-" * len(header))

    for task in TASKS:
        b = load_benchmark(task)
        print(f"# {task}: Bayes ceiling {b.bayes_accuracy:.4f} "
              f"(+/-{b.bayes_accuracy_se:.4f}), "
              f"{len(b.informative_features)} informative features")
        print(f"#   ground truth: {b.explanation}")

        models = {
            "tree": DecisionTreeClassifier(random_state=0),
            "logreg": LogisticRegression(max_iter=3000),
        }
        for name, model in models.items():
            model.fit(b.X_train, b.y_train)
            acc = float(model.score(b.X_test, b.y_test))
            if name == "tree":
                importances = model.feature_importances_
            else:
                importances = np.abs(model.coef_).sum(axis=0)
            scores = b.feature_recovery(importances, top_k=TOP_K)
            results.append({
                "task": task,
                "model": name,
                "accuracy": acc,
                "bayes_accuracy": b.bayes_accuracy,
                "regret": b.accuracy_regret(acc),
                **scores,
            })
            print(
                f"{task:<18} {name:<8} {acc:>7.4f} "
                f"{b.accuracy_regret(acc):>8.4f} "
                f"{scores['spearman']:>9.3f} "
                f"{scores['precision_at_k']:>8.2f} "
                f"{scores['mi_recall_at_k']:>9.3f}"
            )
        print()

    OUT.mkdir(parents=True, exist_ok=True)
    out_json = OUT / "benchmark_eval.json"
    out_json.write_text(json.dumps({"top_k": TOP_K, "results": results}, indent=1))
    print(f"results -> {out_json}")


if __name__ == "__main__":
    main()
