import numpy as np
import pytest

from galois_ml.benchmark import (
    _feature_distribution,
    feature_mutual_information,
    load_benchmark,
)


def test_feature_distribution_is_probability():
    for group in ("C4", "C2^2", "S3", "Q8"):
        for n in (1, 2, 4, 12, 36, 100):
            dist = _feature_distribution(group, n)
            assert abs(sum(dist.values()) - 1.0) < 1e-12
            assert all(p >= 0 for p in dist.values())


def test_quartic_mi_ground_truth():
    mi = feature_mutual_information("quartic", n_coeffs=100)

    def mi_n(n):
        return mi[n - 1]

    # a_1 is constant: zero information.
    assert mi_n(1) == 0.0
    # a_p has the SAME distribution under C4 and C2^2 (both give
    # a_p = 4 with prob 1/4, else 0), so primes carry no information...
    assert mi_n(2) < 1e-12 and mi_n(3) < 1e-12 and mi_n(97) < 1e-12
    # ...but squares do (f=4 is C4-only): the paper's key observation.
    assert mi_n(4) > 0.1 and mi_n(9) > 0.1
    # identical distribution at every prime => identical MI
    assert abs(mi_n(4) - mi_n(9)) < 1e-12
    # composite with a square factor inherits information
    assert mi_n(36) > 0


def test_sextic_mi_orders():
    mi = feature_mutual_information("sextic", n_coeffs=750)

    def mi_n(n):
        return mi[n - 1]

    # a_p and even a_{p^3} have IDENTICAL marginals under C6 and S3
    # (the f=3 vs f=6 distinction is an interaction between a_{p^2}
    # and a_{p^3}, invisible to single-feature MI).
    assert mi_n(2) < 1e-12 and mi_n(5) < 1e-12
    assert mi_n(8) < 1e-12
    # a_{p^2} is informative (f=2 densities 1/6 vs 1/2) ...
    assert mi_n(4) > 0.05
    # ... and a_{p^6} is the top singleton: a_{p^6} = 1 certifies an
    # inert prime, impossible for S3. This is exactly why the paper's
    # sextic decision tree (Fig. 4) splits on a_{2^6} and a_{3^6} first.
    assert mi_n(64) > mi_n(4)
    assert abs(mi_n(64) - mi_n(729)) < 1e-12


def test_benchmark_bundle():
    b = load_benchmark(
        "quartic", n_train=200, n_test=100, n_coeffs=200, bayes_samples=300
    )
    assert b.X_train.shape == (200, 200)
    assert b.X_test.shape == (100, 200)
    assert 0.9 <= b.bayes_accuracy <= 1.0
    assert b.explanation.startswith("Cor 3.3")
    # informative features must include the squares of small primes
    informative = set(b.informative_features.tolist())
    for n in (4, 9, 25, 49):
        assert n - 1 in informative


def test_feature_recovery_scores_ideal_and_bad():
    b = load_benchmark(
        "quartic", n_train=100, n_test=50, n_coeffs=200, bayes_samples=200
    )
    # the exact MI itself must get a perfect score
    ideal = b.feature_recovery(b.feature_mi, top_k=10)
    assert ideal["spearman"] > 0.999
    assert ideal["precision_at_k"] == 1.0
    # adversarial: importance mass on uninformative prime indices
    bad = np.zeros_like(b.feature_mi)
    for n in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29):
        bad[n - 1] = 1.0
    bad_scores = b.feature_recovery(bad, top_k=10)
    assert bad_scores["precision_at_k"] == 0.0


def test_bayes_ceiling_matches_classifier_monte_carlo():
    """The signature-sampling ceiling must agree with running the full
    Bayes classifier on generated zeta coefficients."""
    from galois_ml.benchmark import bayes_ceiling
    from galois_ml.criteria import ChebotarevBayesClassifier
    from galois_ml.data import make_dataset

    task, n_coeffs = "quartic", 200
    fast, se = bayes_ceiling(task, n_coeffs, n_samples=20_000, seed=0)
    X, y, _ = make_dataset(task, 500, n_coeffs, seed=3)
    slow = float(
        np.mean(ChebotarevBayesClassifier(task, n_coeffs).predict(X) == y)
    )
    assert abs(fast - slow) < 0.03  # both are MC estimates
    assert 0 < se < 0.01


def test_export_and_read_roundtrip(tmp_path):
    from galois_ml.benchmark import export_benchmark, read_benchmark

    b = load_benchmark(
        "quartic", n_train=50, n_test=20, n_coeffs=100, bayes_samples=5000
    )
    out = export_benchmark(b, tmp_path / "quartic_n100")
    assert (out / "train.parquet").exists()
    assert (out / "test.parquet").exists()
    assert (out / "metadata.json").exists()

    b2 = read_benchmark(out)
    assert b2.task == b.task
    assert b2.class_names == b.class_names
    assert np.array_equal(b2.X_train, b.X_train)
    assert np.array_equal(b2.y_test, b.y_test)
    assert np.allclose(b2.feature_mi, b.feature_mi)
    assert b2.bayes_accuracy == b.bayes_accuracy
    assert b2.explanation == b.explanation


def test_exported_parquet_is_hf_loadable(tmp_path):
    """The parquet files must be readable as plain tables with the
    documented schema (what datasets.load_dataset('parquet', ...) does)."""
    import pyarrow.parquet as pq

    from galois_ml.benchmark import export_benchmark

    b = load_benchmark(
        "sextic", n_train=30, n_test=10, n_coeffs=100, bayes_samples=5000
    )
    out = export_benchmark(b, tmp_path / "sextic_n100")
    table = pq.read_table(out / "train.parquet")
    assert table.column_names == ["zeta", "label", "group"]
    assert table.num_rows == 30
    row = table.slice(0, 1).to_pylist()[0]
    assert len(row["zeta"]) == 100
    assert row["group"] in b.class_names
    assert row["zeta"][0] == 1.0  # a_1 = 1
