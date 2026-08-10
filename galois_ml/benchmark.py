"""Interpretability benchmarks with provable ground truth.

Most interpretability evaluations lack a reference answer: nobody knows
the "true explanation" for an ImageNet classifier. The Galois tasks are
different - the generative model is fully known, so we can compute:

* the exact Bayes-optimal accuracy (ceiling for any model),
* the exact mutual information I(a_n ; class) of every feature, in
  closed form (the true feature-importance ranking), and
* the proven theorem that constitutes the correct explanation.

This lets us score any classifier (accuracy regret) and any attribution
method (agreement with the true MI ranking) against ground truth.

The MI computation uses the structure of the data model: given the
class, a_n = prod a_{p^k} over the prime powers p^k || n, with
independent factors across primes, and each a_{p^k} takes finitely many
values with Chebotarev probabilities (Cor 2.2 pushes the Frobenius
order distribution through the Euler coefficient formula).
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

from galois_ml.criteria import ChebotarevBayesClassifier
from galois_ml.data import make_dataset
from galois_ml.groups import GROUPS, TASKS
from galois_ml.zeta import _factor_table, euler_coefficient

#: The proven criterion behind each task (the reference "explanation").
GROUND_TRUTH_EXPLANATIONS: dict[str, str] = {
    "quartic": "Cor 3.3 (l=2): Gal = C4 iff a_{p^2}(K) = 0 for some prime p.",
    "nonic": "Cor 3.3 (l=3): Gal = C9 iff a_{p^3}(K) = 0 for some prime p.",
    "sextic": "Cor 3.14/3.15: S3 has no inert primes (no order-6 element); "
              "f=6 signature is a_{p^2} = a_{p^3} = 0.",
    "decic": "Cor 3.20: D5 has no inert primes; f=10 signature is "
             "a_{p^2} = a_{p^5} = 0.",
    "octic_abelian": "Cor 3.24: C8 iff a_{p^4} = 0 for some p; "
                     "C2^3 iff a_{p^2} != 0 for all p.",
    "octic_nonabelian": "Prop 3.33: D4 and Q8 admit identical decomposition "
                        "types; only Chebotarev densities differ "
                        "(f=2 density 5/8 vs 1/8).",
    # held-out tasks: criteria follow from the same theory but appear
    # in no published paper (contamination-resistant).
    "degree14": "HELD OUT. D7 has no order-14 element, hence no inert "
                "primes: f=14 signature is a_{p^2} = 0 and a_{p^7} = 0; "
                "also f=2 density differs (1/14 vs 1/2).",
    "degree21": "HELD OUT. C7:C3 has no order-21 element, hence no inert "
                "primes; f=3 density differs (2/21 vs 14/21).",
    "degree25": "HELD OUT. Cor 3.3 pattern with l=5: Gal = C25 iff "
                "a_{p^5}(K) = 0 for some prime p.",
}


@lru_cache(maxsize=None)
def _prime_power_distribution(
    group_name: str, k: int
) -> tuple[tuple[float, float], ...]:
    """Exact distribution of a_{p^k} for an unramified prime p in a
    Galois extension with the given group (identical for every p).
    Returns ((value, probability), ...)."""
    group = GROUPS[group_name]
    orders, probs = group.frobenius_orders
    dist: dict[float, float] = {}
    for f, prob in zip(orders, probs):
        v = float(euler_coefficient(k, int(f), group.order // int(f)))
        dist[v] = dist.get(v, 0.0) + float(prob)
    return tuple(sorted(dist.items()))


def _feature_distribution(group_name: str, n: int) -> dict[float, float]:
    """Exact distribution of a_n given the class, via independence of
    the prime-power factors of n."""
    if n == 1:
        return {1.0: 1.0}
    _, decomp = _factor_table(n)
    dist: dict[float, float] = {1.0: 1.0}
    m = n
    while m > 1:
        m, p, k = decomp[m]
        combined: dict[float, float] = {}
        for v1, p1 in dist.items():
            for v2, p2 in _prime_power_distribution(group_name, k):
                combined[v1 * v2] = combined.get(v1 * v2, 0.0) + p1 * p2
        dist = combined
    return dist


def _entropy(dist: dict[float, float]) -> float:
    ps = np.array([p for p in dist.values() if p > 0])
    return float(-(ps * np.log2(ps)).sum())


def feature_mutual_information(task: str, n_coeffs: int = 1000) -> np.ndarray:
    """Exact I(a_n ; class) in bits for n = 1..n_coeffs, with the class
    uniform over TASKS[task]. This is the ground-truth feature
    importance ranking for the benchmark."""
    class_names = TASKS[task]
    k = len(class_names)
    mi = np.zeros(n_coeffs)
    for i in range(n_coeffs):
        n = i + 1
        per_class = [_feature_distribution(g, n) for g in class_names]
        mixture: dict[float, float] = {}
        for dist in per_class:
            for v, p in dist.items():
                mixture[v] = mixture.get(v, 0.0) + p / k
        mi[i] = _entropy(mixture) - sum(_entropy(d) for d in per_class) / k
    return np.maximum(mi, 0.0)  # clip float noise


@dataclass
class GaloisBenchmark:
    """A benchmark task with provable ground truth."""

    task: str
    class_names: list[str]
    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    bayes_accuracy: float          # Monte Carlo estimate of the ceiling
    bayes_accuracy_se: float       # its standard error
    feature_mi: np.ndarray         # exact I(a_n; class), n = 1..n_coeffs
    explanation: str               # the proven criterion

    @property
    def informative_features(self) -> np.ndarray:
        """Indices (0-based) of features with nonzero mutual information."""
        return np.where(self.feature_mi > 1e-12)[0]

    # -- scoring -------------------------------------------------------

    def accuracy_regret(self, accuracy: float) -> float:
        """Gap to the Bayes ceiling (values near 0 are optimal; negative
        values are within Monte Carlo noise of the ceiling)."""
        return self.bayes_accuracy - accuracy

    def feature_recovery(
        self, importances: np.ndarray, top_k: int = 20
    ) -> dict[str, float]:
        """Score an attribution vector against the exact MI ground truth.

        Returns:
            spearman: rank correlation with the true MI over all features.
            precision_at_k: fraction of the top-k attributed features
                that are genuinely informative (MI > 0).
            mi_recall_at_k: fraction of total MI captured by the top-k
                attributed features.
        """
        from scipy.stats import spearmanr

        importances = np.asarray(importances, dtype=float)
        if importances.shape != self.feature_mi.shape:
            raise ValueError("importances must have one entry per feature")
        rho = float(spearmanr(importances, self.feature_mi).statistic)
        top = np.argsort(importances)[::-1][:top_k]
        informative = set(self.informative_features.tolist())
        precision = sum(i in informative for i in top) / top_k
        mi_recall = float(self.feature_mi[top].sum() / self.feature_mi.sum())
        return {
            "spearman": rho,
            "precision_at_k": precision,
            "mi_recall_at_k": mi_recall,
        }


def load_benchmark(
    task: str,
    n_train: int = 2400,
    n_test: int = 600,
    n_coeffs: int = 1000,
    seed: int = 20260810,
    bayes_samples: int = 200_000,
) -> GaloisBenchmark:
    """Assemble a benchmark with fixed splits and ground truth. The
    default seed pins the canonical splits for comparability."""
    X, y, class_names = make_dataset(
        task, n_train + n_test, n_coeffs=n_coeffs, seed=seed
    )
    p_hat, se = bayes_ceiling(
        task, n_coeffs, n_samples=bayes_samples, seed=seed + 1
    )
    return GaloisBenchmark(
        task=task,
        class_names=class_names,
        X_train=X[:n_train],
        y_train=y[:n_train],
        X_test=X[n_train:],
        y_test=y[n_train:],
        bayes_accuracy=p_hat,
        bayes_accuracy_se=se,
        feature_mi=feature_mutual_information(task, n_coeffs),
        explanation=GROUND_TRUTH_EXPLANATIONS[task],
    )


def bayes_ceiling(
    task: str,
    n_coeffs: int = 1000,
    n_samples: int = 200_000,
    seed: int = 0,
) -> tuple[float, float]:
    """High-precision Monte Carlo estimate of the Bayes-optimal accuracy.

    Instead of assembling full zeta-coefficient vectors, sample each
    prime's observable signature directly from its class-conditional
    Chebotarev distribution and apply the exact Bayes rule to the
    log-likelihood sums. This is distributionally identical to running
    `ChebotarevBayesClassifier` on `make_dataset` output but orders of
    magnitude faster, allowing sample sizes with standard errors of a
    few 1e-4.

    Returns (accuracy, standard_error).
    """
    clf = ChebotarevBayesClassifier(task, n_coeffs)
    n_classes = len(clf.groups)
    rng = np.random.default_rng(seed)

    # Per prime: unified signature list, class-conditional probabilities,
    # and per-signature log-likelihood vectors (impossible -> -inf).
    per_prime = []
    for p in clf.primes:
        signatures = sorted(
            {s for gi in range(n_classes) for s in clf._signature_density[gi][p]}
        )
        probs = np.array([
            [clf._signature_density[gi][p].get(s, 0.0) for s in signatures]
            for gi in range(n_classes)
        ])  # (n_classes, n_sigs)
        with np.errstate(divide="ignore"):
            ll = np.log(probs.T)  # (n_sigs, n_classes); log 0 = -inf
        per_prime.append((probs, ll))

    correct = 0
    base = n_samples // n_classes
    for true_class in range(n_classes):
        size = base + (1 if true_class < n_samples % n_classes else 0)
        total_ll = np.zeros((size, n_classes))
        for probs, ll in per_prime:
            idx = rng.choice(len(ll), size=size, p=probs[true_class])
            total_ll += ll[idx]
        # -inf - -inf never occurs within a row's argmax comparison;
        # rows where the true class beats others count as correct.
        correct += int(np.sum(np.argmax(total_ll, axis=1) == true_class))

    p_hat = correct / n_samples
    se = float(np.sqrt(p_hat * (1 - p_hat) / n_samples))
    return float(p_hat), se


# -- Hugging Face export (parquet + JSON) ----------------------------------

BENCHMARK_FORMAT_VERSION = "1.0"


def export_benchmark(benchmark: GaloisBenchmark, out_dir: str | Path) -> Path:
    """Write a benchmark config in Hugging Face-native formats:

    out_dir/
      train.parquet    columns: zeta (list<float64>), label (int64),
      test.parquet              group (string)
      metadata.json    ceiling, exact feature MI, explanation, classes

    The parquet files load directly with `datasets.load_dataset
    ("parquet", data_files=...)`; metadata.json carries the ground
    truth that makes this a benchmark rather than just a dataset.
    """
    import json

    import pyarrow as pa
    import pyarrow.parquet as pq

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    def _write_split(X: np.ndarray, y: np.ndarray, name: str) -> None:
        table = pa.table({
            "zeta": pa.array(X.tolist(), type=pa.list_(pa.float64())),
            "label": pa.array(y.tolist(), type=pa.int64()),
            "group": pa.array(
                [benchmark.class_names[i] for i in y], type=pa.string()
            ),
        })
        pq.write_table(table, out / f"{name}.parquet")

    _write_split(benchmark.X_train, benchmark.y_train, "train")
    _write_split(benchmark.X_test, benchmark.y_test, "test")

    metadata = {
        "format_version": BENCHMARK_FORMAT_VERSION,
        "task": benchmark.task,
        "class_names": benchmark.class_names,
        "n_coeffs": int(benchmark.X_train.shape[1]),
        "n_train": int(len(benchmark.y_train)),
        "n_test": int(len(benchmark.y_test)),
        "bayes_accuracy": benchmark.bayes_accuracy,
        "bayes_accuracy_se": benchmark.bayes_accuracy_se,
        "explanation": benchmark.explanation,
        "feature_names": [f"a_{n}" for n in range(1, benchmark.X_train.shape[1] + 1)],
        "feature_mutual_information": benchmark.feature_mi.tolist(),
        "source": "Lee & Lee, arXiv:2508.06670; synthetic Chebotarev model",
    }
    (out / "metadata.json").write_text(json.dumps(metadata, indent=1))
    return out


def read_benchmark(in_dir: str | Path) -> GaloisBenchmark:
    """Load a benchmark config previously written by `export_benchmark`."""
    import json

    import pyarrow.parquet as pq

    src = Path(in_dir)
    meta = json.loads((src / "metadata.json").read_text())

    def _read_split(name: str) -> tuple[np.ndarray, np.ndarray]:
        table = pq.read_table(src / f"{name}.parquet")
        X = np.array(table.column("zeta").to_pylist(), dtype=np.float64)
        y = np.array(table.column("label").to_pylist(), dtype=np.int64)
        return X, y

    X_train, y_train = _read_split("train")
    X_test, y_test = _read_split("test")
    return GaloisBenchmark(
        task=meta["task"],
        class_names=meta["class_names"],
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        bayes_accuracy=meta["bayes_accuracy"],
        bayes_accuracy_se=meta["bayes_accuracy_se"],
        feature_mi=np.array(meta["feature_mutual_information"]),
        explanation=meta["explanation"],
    )
