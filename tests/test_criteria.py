import numpy as np
import pytest

from galois_ml.criteria import ChebotarevBayesClassifier, TheoremClassifier
from galois_ml.data import make_dataset

# Minimum accuracies: these sit just under the Bayes-optimal ceiling of
# the unramified Chebotarev data model with N = 1000 coefficients. The
# ceiling is below 1.0 for tasks whose distinguishing Frobenius classes
# are only visible at the few primes p with p^k <= N (e.g. sextic C6 vs
# S3 hinges on 11 primes with visible a_{p^2}); it rises with N.
THRESHOLDS = {
    "quartic": 0.999,          # Cor 3.3 is (effectively) exact here
    "nonic": 0.99,             # inert primes have density 2/3 for C9
    "octic_abelian": 0.95,
    "octic_nonabelian": 0.95,
    "sextic": 0.92,
    "decic": 0.93,
}


@pytest.mark.parametrize("task", sorted(THRESHOLDS))
def test_theorem_classifier_accuracy(task):
    X, y, _ = make_dataset(task, n_samples=400, n_coeffs=1000, seed=42)
    clf = TheoremClassifier(task, n_coeffs=1000)
    accuracy = float(np.mean(clf.predict(X) == y))
    assert accuracy >= THRESHOLDS[task], f"{task}: accuracy {accuracy:.4f}"


@pytest.mark.parametrize("task", sorted(THRESHOLDS))
def test_theorem_matches_bayes(task):
    # The proven deterministic rules are zero-density exclusions, so the
    # theorem classifier should agree with the exact Bayes rule.
    X, y, _ = make_dataset(task, n_samples=200, n_coeffs=1000, seed=7)
    thm = TheoremClassifier(task).predict(X)
    bayes = ChebotarevBayesClassifier(task).predict(X)
    assert np.mean(thm == bayes) >= 0.99


def test_dataset_shapes_and_labels():
    X, y, names = make_dataset("quartic", n_samples=10, n_coeffs=50, seed=0)
    assert X.shape == (10, 50)
    assert set(np.unique(y)) <= {0, 1}
    assert names == ["C4", "C2^2"]
    assert np.all(X[:, 0] == 1)  # a_1 = 1 always
