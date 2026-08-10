"""Classify Galois groups of *real* number fields from the LMFDB.

Downloads defining polynomials via the LMFDB API, computes zeta
coefficients by factoring mod p (Dedekind's theorem), and evaluates the
theorem-backed classifier and a decision tree - closing the loop between
the synthetic Chebotarev model and genuine arithmetic data, ramified
primes included.

Requires network access on first run; responses are cached under
.lmfdb_cache/.

Run:  python experiments/classify_lmfdb_fields.py [task ...]
"""

from __future__ import annotations

import sys

import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier

from galois_ml.criteria import TheoremClassifier
from galois_ml.lmfdb import make_lmfdb_dataset

DEFAULT_TASKS = ["quartic", "nonic"]
PER_CLASS = 100
N_COEFFS = 1000


def main() -> None:
    tasks = sys.argv[1:] or DEFAULT_TASKS
    for task in tasks:
        print(f"\n=== {task} (LMFDB, {PER_CLASS}/class, N={N_COEFFS}) ===")
        X, y, names = make_lmfdb_dataset(
            task, per_class=PER_CLASS, n_coeffs=N_COEFFS
        )
        print(f"classes: {names}, samples: {len(y)}")

        thm = TheoremClassifier(task, n_coeffs=N_COEFFS)
        thm_acc = float(np.mean(thm.predict(X) == y))
        print(f"theorem classifier accuracy: {thm_acc:.4f}")

        tree = DecisionTreeClassifier(random_state=0)
        cv = cross_val_score(tree, X, y, cv=5)
        print(f"decision tree 5-fold CV accuracy: "
              f"{cv.mean():.4f} +/- {cv.std():.3f}")


if __name__ == "__main__":
    main()
