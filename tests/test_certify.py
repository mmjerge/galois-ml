"""Tests for certified model auditing (galois_ml.certify).

The claims that matter: the exact posterior really is the optimal
predictor (so a model cannot beat it), calibration is zero exactly when a
model reproduces it, admissibility never rejects honest data, and
certified perturbations really do preserve the true label.
"""

import numpy as np
import pytest

from galois_ml.benchmark import load_benchmark
from galois_ml.certify import (
    admissible_mask,
    calibration_report,
    certified_perturbation,
    exact_posterior,
)
from galois_ml.groups import GROUPS


@pytest.fixture(scope="module")
def bench():
    return load_benchmark(
        "quartic", n_train=200, n_test=150, n_coeffs=200, bayes_samples=5000
    )


def test_posterior_is_a_distribution(bench):
    post = exact_posterior("quartic", bench.X_test)
    assert post.shape == (len(bench.y_test), len(bench.class_names))
    assert np.allclose(post.sum(axis=1), 1.0)
    assert (post >= 0).all()


def test_posterior_argmax_is_the_ceiling(bench):
    """The exact posterior must be at least as accurate as any model on
    the same data -- it is the Bayes rule by construction."""
    from sklearn.tree import DecisionTreeClassifier

    post = exact_posterior("quartic", bench.X_test)
    bayes_acc = float(np.mean(np.argmax(post, axis=1) == bench.y_test))
    tree = DecisionTreeClassifier(random_state=0).fit(
        bench.X_train, bench.y_train
    )
    tree_acc = float(np.mean(tree.predict(bench.X_test) == bench.y_test))
    assert bayes_acc >= tree_acc - 1e-12


def test_perfect_model_has_zero_excess_loss(bench):
    """A model that reports the true posterior pays exactly the
    irreducible cost and nothing more."""
    post = exact_posterior("quartic", bench.X_test)
    rep = calibration_report(post, post, bench.y_test)
    assert rep.mean_kl == pytest.approx(0.0, abs=1e-9)
    assert rep.excess_log_loss == pytest.approx(0.0, abs=1e-9)
    assert rep.exact_calibration_error == pytest.approx(0.0, abs=1e-9)
    assert rep.model_log_loss == pytest.approx(rep.optimal_log_loss, abs=1e-9)


def test_excess_log_loss_equals_mean_kl(bench):
    """The decomposition must hold exactly: everything above the true
    posterior's entropy is attributable to the model."""
    post = exact_posterior("quartic", bench.X_test)
    rng = np.random.default_rng(0)
    noisy = np.clip(post + rng.normal(0, 0.15, post.shape), 1e-6, None)
    noisy /= noisy.sum(axis=1, keepdims=True)
    rep = calibration_report(noisy, post, bench.y_test)
    assert rep.excess_log_loss == pytest.approx(rep.mean_kl, abs=1e-9)
    assert rep.mean_kl > 0


def test_overconfidence_is_detected(bench):
    """A sharpened model is flagged as overconfident relative to truth,
    which binned ECE can miss when accuracy is high."""
    post = exact_posterior("quartic", bench.X_test)
    sharp = post ** 4
    sharp /= sharp.sum(axis=1, keepdims=True)
    rep = calibration_report(sharp, post, bench.y_test)
    # only meaningful where the truth is not already deterministic
    if rep.optimal_log_loss > 1e-6:
        assert rep.overconfidence > 0
        assert rep.overconfident_fraction > 0


def test_admissibility_never_rejects_honest_data(bench):
    """Soundness: the whole point of a decidable predicate."""
    degree = GROUPS[bench.class_names[0]].order
    assert admissible_mask(bench.X_train, degree).all()
    assert admissible_mask(bench.X_test, degree).all()


def test_admissibility_rejects_impossible_values(bench):
    degree = GROUPS[bench.class_names[0]].order
    X = bench.X_test.copy()
    X[0, 3] = 7      # a_4 = 7 is not attainable in degree 4
    X[1, 1] = -1     # negative counts are impossible
    X[2, 8] = 2.5    # non-integral
    mask = admissible_mask(X, degree)
    assert not mask[0] and not mask[1] and not mask[2]
    assert mask[3:].all()


def test_certified_perturbation_preserves_the_label(bench):
    """The label is preserved by construction: perturbed rows must still
    be classified correctly by the Bayes rule at the same rate, and must
    remain admissible."""
    degree = GROUPS[bench.class_names[0]].order
    X2 = certified_perturbation(
        bench.X_test, "quartic", class_index=bench.y_test, seed=1
    )
    assert X2.shape == bench.X_test.shape
    assert not np.allclose(X2, bench.X_test)  # something actually changed
    assert admissible_mask(X2, degree).all()  # still legal inputs
    # the Bayes rule on resampled data is still near the ceiling: the
    # transformation moved the evidence, not the truth
    post = exact_posterior("quartic", X2)
    acc = float(np.mean(np.argmax(post, axis=1) == bench.y_test))
    assert acc > 0.9


def test_certified_perturbation_requires_class_index(bench):
    with pytest.raises(ValueError):
        certified_perturbation(bench.X_test, "quartic")


def test_inadmissible_rows_fall_back_to_prior():
    """A row no class can explain gets the prior, not a fabricated
    posterior, and the count is recorded."""
    X = np.zeros((1, 50))
    X[0, 0] = 1.0
    X[0, 3] = 7.0  # impossible a_4
    post = exact_posterior("quartic", X)
    assert np.allclose(post.sum(), 1.0)
    rep = calibration_report(post, post, inadmissible=1)
    assert rep.inadmissible_rows == 1


# -- regression tests for external-review counterexamples --------------------

def test_perturbation_preserves_multiplicativity():
    """Review counterexample: resampling prime 2 changed a_2 but left
    a_6 = 36, so the output belonged to no valid law. Composites must be
    rebuilt from the updated prime-power values."""
    X = np.zeros((1, 100))
    X[0, 0] = 1.0
    X[0, 1] = 6.0   # a_2, prime 2 split completely
    X[0, 2] = 6.0   # a_3
    X[0, 5] = 36.0  # a_6 = a_2 * a_3
    for seed in range(30):
        X2 = certified_perturbation(
            X, "sextic", primes=[2], class_index=np.array([0]), seed=seed
        )
        assert np.isclose(X2[0, 5], X2[0, 1] * X2[0, 2]), (
            f"seed {seed}: a_6={X2[0,5]} != a_2*a_3={X2[0,1]*X2[0,2]}"
        )


def test_perturbation_validates_class_index(bench):
    with pytest.raises(ValueError):
        certified_perturbation(
            bench.X_test, "quartic", class_index=bench.y_test[:-1]  # short
        )
    with pytest.raises(ValueError):
        certified_perturbation(
            bench.X_test, "quartic",
            class_index=np.full(len(bench.X_test), -1),  # negative
        )


def test_joint_admissibility_rejects_same_prime_contradiction(bench):
    """a_2=4 (degree 4, f=1) forces a_4=10; a_4=2 is individually
    attainable (f=2) but jointly impossible. Coordinatewise screening
    passed this; the joint decision must not."""
    degree = GROUPS[bench.class_names[0]].order
    row = bench.X_test[0].copy()
    X = np.vstack([row])
    X[0, 1] = 4.0   # a_2: split completely
    X[0, 3] = 2.0   # a_4: inconsistent with f=1 (would be 10)
    assert not admissible_mask(X, degree)[0]


def test_joint_admissibility_rejects_multiplicativity_break(bench):
    degree = GROUPS[bench.class_names[0]].order
    X = bench.X_test[:1].copy()
    correct = X[0, 1] * X[0, 2]
    # coordinatewise-attainable but wrong product
    forged = 16.0 if not np.isclose(correct, 16.0) else 0.0
    X[0, 5] = forged
    assert not admissible_mask(X, degree)[0]


def test_admissibility_rejects_nonfinite(bench):
    degree = GROUPS[bench.class_names[0]].order
    X = bench.X_test[:1].copy()
    X[0, 8] = np.inf
    assert not admissible_mask(X, degree)[0]  # must not crash


def test_zero_prior_excludes_class():
    """Review 5.1: replacing zero prior mass with tiny positive mass let
    an excluded class win. Zero prior must mean zero posterior."""
    X = np.zeros((1, 50))
    X[0, 0] = 1.0
    post = exact_posterior("quartic", X, prior=np.array([1.0, 0.0]))
    assert post[0, 1] == 0.0
    with pytest.raises(ValueError):
        exact_posterior("quartic", X, prior=np.array([-0.5, 1.5]))
    with pytest.raises(ValueError):
        exact_posterior("quartic", X, prior=np.array([0.0, 0.0]))


def test_return_valid_flags_undefined_posteriors():
    X = np.zeros((2, 50))
    X[:, 0] = 1.0
    X[0, 3] = 7.0  # impossible a_4: no class explains it
    post, valid = exact_posterior("quartic", X, return_valid=True)
    assert not valid[0]
    assert np.allclose(post.sum(axis=1), 1.0)


def test_overconfidence_at_model_choice():
    """Review counterexample: Q=(0.9,0.1), P=(0.1,0.9) gave zero
    'overconfidence' under max-vs-max; at the model's chosen class the
    gap is 0.9 - 0.1 = 0.8."""
    Q = np.array([[0.9, 0.1]])
    P = np.array([[0.1, 0.9]])
    rep = calibration_report(P, Q)
    assert rep.overconfidence == pytest.approx(0.8, abs=1e-9)
    assert rep.overconfident_fraction == 1.0


def test_expected_accuracy_when_labels_absent():
    """With y=None the report must give exact conditional expectations,
    not agreement-with-argmax (which reported bayes accuracy 1.0)."""
    Q = np.array([[0.7, 0.3], [0.6, 0.4]])
    P = np.array([[0.2, 0.8], [0.9, 0.1]])  # first prediction is wrong
    rep = calibration_report(P, Q)
    assert rep.bayes_accuracy_on_sample == pytest.approx(0.65)  # mean max Q
    assert rep.accuracy == pytest.approx((0.3 + 0.6) / 2)  # mean Q at choice
    assert rep.accuracy_regret == pytest.approx(0.65 - 0.45)
    assert rep.accuracy_regret >= 0


def test_support_violations_counted():
    Q = np.array([[0.5, 0.5]])
    P = np.array([[1.0, 0.0]])  # zero mass on a class with Q > 0
    rep = calibration_report(P, Q)
    assert rep.support_violations == 1
    assert rep.eps > 0
