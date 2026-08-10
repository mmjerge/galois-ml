"""Interpretable ML baselines reproducing the experiments of the paper:
decision trees and logistic regression on zeta coefficients, compared
against the theorem-backed classifier.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from galois_ml.criteria import TheoremClassifier
from galois_ml.data import make_dataset
from galois_ml.zeta import _factor_table


def describe_index(idx: int) -> str:
    """Human-readable description of feature `idx` (which is a_{idx+1})."""
    n = idx + 1
    if n == 1:
        return "a_1"
    _, decomp = _factor_table(n)
    factors = []
    m = n
    while m > 1:
        rest, p, a = decomp[m]
        factors.append(f"{p}^{a}" if a > 1 else f"{p}")
        m = rest
    return f"a_{n} = a_({' * '.join(reversed(factors))})"


@dataclass
class TaskResult:
    task: str
    class_names: list[str]
    accuracy: dict[str, float] = field(default_factory=dict)
    top_tree_features: list[str] = field(default_factory=list)


def run_task(
    task: str,
    n_samples: int = 3000,
    n_coeffs: int = 1000,
    seed: int = 0,
    test_size: float = 0.2,
) -> TaskResult:
    """Train DT + LR on a synthetic zeta-coefficient dataset and compare
    with the theorem-backed classifier. Mirrors Section 3 of the paper
    (80/20 split, default scikit-learn hyperparameters)."""
    X, y, class_names = make_dataset(task, n_samples, n_coeffs, seed=seed)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )
    result = TaskResult(task=task, class_names=class_names)

    tree = DecisionTreeClassifier(random_state=seed)
    tree.fit(X_tr, y_tr)
    result.accuracy["decision_tree"] = float(tree.score(X_te, y_te))
    importances = tree.feature_importances_
    top = np.argsort(importances)[::-1][:8]
    result.top_tree_features = [
        f"{describe_index(i)} (importance {importances[i]:.3f})"
        for i in top
        if importances[i] > 0
    ]

    lr = LogisticRegression(max_iter=3000)
    lr.fit(X_tr, y_tr)
    result.accuracy["logistic_regression"] = float(lr.score(X_te, y_te))

    theorem = TheoremClassifier(task, n_coeffs=n_coeffs)
    result.accuracy["theorem"] = float(
        np.mean(theorem.predict(X_te) == y_te)
    )
    return result


def run_all(
    tasks: list[str] | None = None, **kwargs
) -> list[TaskResult]:
    from galois_ml.groups import TASKS

    return [run_task(t, **kwargs) for t in (tasks or list(TASKS))]
