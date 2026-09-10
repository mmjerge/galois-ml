"""Certified model auditing: scoring a model against a per-input
ground-truth posterior rather than against a proxy.

The security half of this repository (`galois_ml.guard`,
`galois_ml.tool_poisoning`) measures whether a pipeline's *evidence* can
be trusted. This module measures whether a *model* can be, at the level
where trustworthy-ML claims are normally weakest: calibration,
out-of-distribution rejection, feature attribution, and invariance are
all usually evaluated against proxies, because the true conditional
distribution of the label is unknown.

Here it is known. `ChebotarevBayesClassifier.log_likelihoods` gives the
exact class-conditional likelihood of any input under the generative
law, so with a stated prior the posterior P(class | x) is available for
every individual input. That upgrades four standard audits:

* **Calibration.** Instead of binning predictions and using observed
  frequency as a stand-in for truth, compare the model's distribution
  to the true posterior directly. The excess log loss is exactly
  KL(true || model) in bits, and it decomposes: the entropy of the true
  posterior is irreducible, everything above it is the model's fault.
  No binning, no bin-width sensitivity, and it is defined per example.
* **Admissibility.** The generative law bounds which feature vectors can
  occur at all. Rejecting the rest is decidable and sound by
  construction, unlike a score threshold. (Related to, but stronger
  than, the tool guard's checks: this mask decides full joint
  admissibility of a row, while the guard's marginal-output checks are
  a per-coordinate necessary condition.)
* **Attribution.** Exact per-feature MARGINAL mutual information is
  available (`GaloisBenchmark.feature_recovery`). Importance mass on
  MI=0 features is a descriptive statistic, NOT certified spurious
  reliance: marginal independence does not imply conditional
  irrelevance (XOR is the standard counterexample), and these features
  have substantial deterministic dependence.
* **Invariance.** Resampling a prime's observable signature from the
  same class-conditional distribution (then restoring
  multiplicativity) preserves the latent label by construction. A flip
  is a defect only under the Bayes-label-stable filter described at
  `certified_perturbation`; unconditional flips can be legitimate
  evidence-tracking or even corrections.

A note on what this does and does not establish. The posterior is exact
*for the generative model of `galois_ml.data`* (unramified primes,
Chebotarev distribution, independence across primes). That model is
validated against real number fields via LMFDB rather than assumed, but
a calibration number computed here is a statement about behaviour on
that law, not a guarantee about arbitrary deployment data. The honest
claim is that the audit itself is exact, which is what lets a
disagreement between two models be attributed to the models.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

BITS = np.log(2.0)


def exact_posterior(
    task: str,
    X: np.ndarray,
    n_coeffs: int | None = None,
    prior: np.ndarray | None = None,
    return_valid: bool = False,
):
    """Exact P(class | x) for each row of X, under the generative law.

    Args:
        task: benchmark task name.
        X: (n, d) feature matrix of zeta coefficients a_1..a_d.
        n_coeffs: coefficients to use (defaults to X's width).
        prior: class prior; uniform if omitted. Must be nonnegative,
            finite, and have positive total; zero entries genuinely
            exclude a class (represented as -inf log prior, not as tiny
            positive mass, which would let an excluded class win on an
            observation only it can explain).
        return_valid: also return a boolean mask of rows for which the
            posterior is actually defined.

    Returns:
        (n, k) array of posteriors, or (posteriors, valid) if
        `return_valid`. Rows whose observations have zero marginal
        probability under the prior (impossible under every candidate
        with prior mass) have NO defined posterior; those rows carry the
        prior as a placeholder and `valid=False`. Do not score them as
        posteriors: `admissible_mask` identifies them independently.
    """
    from galois_ml.criteria import ChebotarevBayesClassifier

    X = np.atleast_2d(np.asarray(X, dtype=float))
    clf = ChebotarevBayesClassifier(task, n_coeffs or X.shape[1])
    k = len(clf.groups)
    if prior is None:
        prior = np.full(k, 1.0 / k)
    prior = np.asarray(prior, dtype=float)
    if (prior.shape != (k,) or not np.all(np.isfinite(prior))
            or np.any(prior < 0) or prior.sum() <= 0):
        raise ValueError(f"prior must be {k} nonnegative finite values "
                         f"with positive total")
    prior = prior / prior.sum()
    with np.errstate(divide="ignore"):
        log_prior = np.where(prior > 0, np.log(np.where(prior > 0, prior, 1)),
                             -np.inf)

    out = np.empty((len(X), k))
    valid = np.ones(len(X), dtype=bool)
    for i, row in enumerate(X):
        # Validity must be checked directly: `log_likelihoods` SKIPS
        # prime signatures impossible under every candidate (a deliberate
        # robustness choice for real number fields, where such primes are
        # ramified), so an impossible observation silently contributes
        # nothing instead of producing -inf. External review asked
        # whether unsupported observations were handled; this is where.
        if not _row_in_mixture_support(clf, row):
            out[i] = prior  # placeholder, NOT a posterior
            valid[i] = False
            continue
        logits = clf.log_likelihoods(row) + log_prior
        if not np.any(np.isfinite(logits)):
            out[i] = prior
            valid[i] = False
            continue
        m = np.max(logits[np.isfinite(logits)])
        w = np.exp(np.where(np.isfinite(logits), logits - m, -np.inf))
        out[i] = w / w.sum()
    return (out, valid) if return_valid else out


def _row_in_mixture_support(clf, row: np.ndarray) -> bool:
    """Does the candidate mixture assign this row positive probability?

    True iff SOME SINGLE candidate group assigns positive density to
    every prime's visible signature (the quantifier matters: per-prime
    existence of possibly different groups is a strictly weaker
    condition, flagged by external review -- with the current task
    roster the two coincide because each task has a class whose element
    orders contain every other's, but the correct predicate should not
    depend on that accident), all prime-power entries are finite
    nonnegative integers, a_1 = 1, and composite coefficients are
    multiplicative.
    Rows failing this have P(x) = 0 under the synthetic law, hence no
    defined posterior. (Genuine number fields with ramified primes will
    also fail: validity is relative to the unramified generator.)
    """
    from galois_ml.zeta import _factor_table

    if len(row) >= 1 and row[0] != 1.0:
        return False
    class_ok = [True] * len(clf.groups)
    for p in clf.primes:
        vals = [row[p**k - 1] for k in range(1, clf._max_power[p] + 1)]
        if any(not np.isfinite(v) or v != int(v) or v < 0 for v in vals):
            return False
        sig = tuple(int(v) for v in vals)
        for gi in range(len(clf.groups)):
            if clf._signature_density[gi][p].get(sig, 0.0) == 0.0:
                class_ok[gi] = False
        if not any(class_ok):
            return False
    _, decomp = _factor_table(len(row))
    for n in range(2, len(row) + 1):
        m, p, a = decomp[n]
        if m == 1:
            continue
        if row[n - 1] != row[m - 1] * row[p**a - 1]:
            return False
    return True


def admissible_mask(X: np.ndarray, degree: int) -> np.ndarray:
    """Which rows are in the support of the unramified degree-d law.

    This is a full joint decision, not a per-coordinate screen (external
    review showed the earlier coordinatewise version accepted rows like
    a_2=6 with a_4=3, individually attainable but jointly impossible).
    Under the unramified model a row is admissible iff:

    * a_1 = 1 and every entry is finite;
    * for each prime p, SOME divisor f of the degree reproduces every
      a_(p^k) in the row (all powers of one prime share one residue
      degree);
    * every composite coefficient equals the product of its prime-power
      parts (multiplicativity).

    Because each prime's residue degree is chosen independently in the
    generative law, these conditions are jointly sufficient as well as
    necessary: a passing row is realized by an explicit f-assignment.
    Still class-agnostic (f ranges over divisors, not any candidate's
    element orders) and still scoped to the unramified model -- genuine
    number fields' ramified primes are out of scope.

    Returns a boolean array; True = admissible.
    """
    from galois_ml.zeta import _factor_table, euler_coefficient

    X = np.atleast_2d(np.asarray(X, dtype=float))
    dim = X.shape[1]
    _, decomp = _factor_table(dim)
    divisors = [f for f in range(1, degree + 1) if degree % f == 0]
    primes = [n for n in range(2, dim + 1)
              if decomp[n][0] == 1 and decomp[n][2] == 1]

    ok = np.ones(len(X), dtype=bool)
    for i, row in enumerate(X):
        if not np.all(np.isfinite(row)) or (dim >= 1 and row[0] != 1.0):
            ok[i] = False
            continue
        good = True
        for p in primes:
            ks = []
            pk = p
            while pk <= dim:
                ks.append((pk, len(ks) + 1))
                pk *= p
            if not any(
                all(row[idx - 1] == euler_coefficient(k, f, degree // f)
                    for idx, k in ks)
                for f in divisors
            ):
                good = False
                break
        if good:
            for n in range(2, dim + 1):
                m, p, a = decomp[n]
                if m == 1:
                    continue
                if row[n - 1] != row[m - 1] * row[p**a - 1]:
                    good = False
                    break
        ok[i] = good
    return ok


@dataclass
class CalibrationReport:
    """Model behaviour against the exact posterior. All in bits."""

    n: int
    model_log_loss: float        # cross entropy of the model, bits
    optimal_log_loss: float      # entropy of the true posterior, bits
    excess_log_loss: float       # model - optimal; equals mean KL below
    mean_kl: float               # mean KL(true || model), bits
    median_kl: float
    max_kl: float
    exact_calibration_error: float   # mean |p_model - p_true| on truth
    overconfidence: float        # mean (conf_model - conf_true), signed
    overconfident_fraction: float    # fraction of rows more confident
    accuracy: float
    bayes_accuracy_on_sample: float  # argmax of the true posterior
    accuracy_regret: float
    inadmissible_rows: int
    eps: float = 1e-12               # smoothing applied before KL
    support_violations: int = 0      # rows where the model put ~0 mass
                                     # on a class with positive true
                                     # posterior (KL is then eps-dependent)

    def to_dict(self) -> dict:
        return asdict(self)


def calibration_report(
    probs: np.ndarray,
    true_posterior: np.ndarray,
    y: np.ndarray | None = None,
    inadmissible: int = 0,
    eps: float = 1e-12,
) -> CalibrationReport:
    """Score predicted distributions against the exact posterior.

    The headline quantity is KL(true || model) in bits: the expected
    excess log loss the model pays for disagreeing with the law. It is
    zero exactly when the model reproduces the posterior, needs no bin
    width, and is defined per example. Name it precisely when reporting:
    this measures POSTERIOR FIDELITY, which is strictly stronger than
    calibration (a model that always predicts the class prior is
    perfectly calibrated yet has positive KL here), so it is a
    complement to ECE, not a drop-in replacement.

    Models that assign exactly zero probability to a class with positive
    true posterior have infinite KL; the `eps` clip makes the reported
    number "KL to the eps-smoothed model" instead. `support_violations`
    counts the affected rows so an eps-sensitive number cannot pass as
    an exact one.

    Direction matters and is chosen deliberately: KL(true || model)
    penalizes assigning low probability to outcomes that are genuinely
    likely, which is the failure mode that matters for a model whose
    output is acted upon.

    Args:
        probs: (n, k) predicted class distributions.
        true_posterior: (n, k) exact posteriors from `exact_posterior`.
        y: optional realized labels, for accuracy alongside calibration.
        inadmissible: count of rows excluded as inadmissible, recorded
            for provenance so a filtered audit is not mistaken for a
            complete one.
    """
    P = np.clip(np.asarray(probs, dtype=float), eps, 1.0)
    P = P / P.sum(axis=1, keepdims=True)
    Q = np.asarray(true_posterior, dtype=float)
    if P.shape != Q.shape:
        raise ValueError("probs and true_posterior must have the same shape")

    # KL(true || model), per row, in bits
    Q_safe = np.where(Q > 0, Q, 1.0)  # see note on 0 log 0 below
    terms = np.where(Q > 0, Q * np.log(Q_safe / P), 0.0)
    kl = terms.sum(axis=1) / BITS

    # log(0) is only ever multiplied by Q = 0 here (0 log 0 = 0), but
    # numpy evaluates both branches of `where`, so guard the argument
    Qs = np.where(Q > 0, Q, 1.0)
    opt_ll = float(np.mean(np.where(Q > 0, -Q * np.log(Qs), 0.0).sum(axis=1))
                   / BITS)
    model_ll = float(np.mean((-Q * np.log(P)).sum(axis=1)) / BITS)

    true_arg = np.argmax(Q, axis=1)
    p_true_on_truth = Q[np.arange(len(Q)), true_arg]
    p_model_on_truth = P[np.arange(len(P)), true_arg]
    # Overconfidence is judged at the MODEL's chosen class: P_c - Q_c for
    # c = argmax P. Comparing max(P) against max(Q) hid the failure mode
    # that matters (review counterexample: Q=(0.9,0.1), P=(0.1,0.9) gave
    # zero "overconfidence" while the model put 0.9 on an event whose
    # true conditional probability is 0.1).
    model_arg = np.argmax(P, axis=1)
    conf_model = P[np.arange(len(P)), model_arg]
    conf_true = Q[np.arange(len(Q)), model_arg]

    if y is not None:
        y = np.asarray(y)
        acc = float(np.mean(model_arg == y))
        bayes_acc = float(np.mean(true_arg == y))
    else:
        # Exact conditional expectations under the law, given these
        # inputs: E[accuracy] = mean Q(model's choice), Bayes =
        # mean max Q. Their difference is exact conditional expected
        # 0-1 regret and is nonnegative by construction, unlike the
        # earlier agreement-with-argmax proxy.
        acc = float(np.mean(Q[np.arange(len(Q)), model_arg]))
        bayes_acc = float(np.mean(Q.max(axis=1)))

    support_viol = int(np.sum(np.any(
        (np.asarray(probs, dtype=float) < eps) & (Q > 0), axis=1
    )))

    return CalibrationReport(
        n=len(P),
        model_log_loss=model_ll,
        optimal_log_loss=opt_ll,
        excess_log_loss=model_ll - opt_ll,
        mean_kl=float(np.mean(kl)),
        median_kl=float(np.median(kl)),
        max_kl=float(np.max(kl)),
        exact_calibration_error=float(
            np.mean(np.abs(p_model_on_truth - p_true_on_truth))
        ),
        overconfidence=float(np.mean(conf_model - conf_true)),
        overconfident_fraction=float(np.mean(conf_model > conf_true)),
        accuracy=acc,
        bayes_accuracy_on_sample=bayes_acc,
        accuracy_regret=bayes_acc - acc,
        inadmissible_rows=int(inadmissible),
        eps=float(eps),
        support_violations=support_viol,
    )


def certified_perturbation(
    X: np.ndarray,
    task: str,
    primes: list[int] | None = None,
    n_coeffs: int | None = None,
    seed: int = 0,
    class_index: np.ndarray | None = None,
) -> np.ndarray:
    """Resample selected primes' signatures from the SAME class, then
    rebuild composite coefficients, so the output is a genuine sample of
    the same class-conditional law.

    The generative law draws each prime's inertia degree independently
    from its class's Chebotarev measure. Redrawing one prime's signature
    from that same measure (and restoring multiplicativity at composite
    indices) produces another input from exactly the same law: the
    latent label is preserved by construction, with no appeal to
    imperceptibility.

    What a prediction flip means, stated carefully (per external
    review): resampling changes the *evidence*, so even a Bayes-optimal
    classifier may legitimately flip on ambiguous inputs, and a flip can
    also *correct* an initially wrong prediction. The defensible metric
    is Bayes-label-stable resampling degradation: count flips only where
    (a) the same class wins the true posterior at both endpoints and
    (b) the original prediction agreed with it. `certify_eval.py`
    implements the posterior-stability filter; interpret rates as
    stability under evidence resampling, not as unconditional certified
    invariance.

    Args:
        class_index: per-row class to resample from. Required, since
            resampling from the wrong class would not preserve the label.
    """
    from galois_ml.criteria import ChebotarevBayesClassifier
    from galois_ml.zeta import _factor_table, euler_coefficient

    X = np.atleast_2d(np.asarray(X, dtype=float)).copy()
    if class_index is None:
        raise ValueError("class_index is required to preserve the label")
    class_index = np.asarray(class_index, dtype=int)
    if class_index.shape != (len(X),):
        raise ValueError("class_index must have exactly one entry per row")
    clf = ChebotarevBayesClassifier(task, n_coeffs or X.shape[1])
    if np.any(class_index < 0) or np.any(class_index >= len(clf.groups)):
        raise ValueError("class_index entries must be valid class labels")
    rng = np.random.default_rng(seed)
    targets = primes if primes is not None else clf.primes[:5]
    dim = X.shape[1]
    _, decomp = _factor_table(dim)

    for i, row_class in enumerate(class_index):
        group = clf.groups[row_class]
        orders, probs = group.frobenius_orders
        for p in targets:
            if p > dim:
                continue
            f = int(rng.choice(np.asarray(orders), p=np.asarray(probs)))
            g = group.order // f
            # every power of p present in the feature window, not just
            # those the classifier happens to track
            pk, k = p, 1
            while pk <= dim:
                X[i, pk - 1] = euler_coefficient(k, f, g)
                pk *= p
                k += 1
        # Rebuild every composite coefficient from the (updated)
        # prime-power values. Without this, resampling a_2 left a_6 at
        # its old value, breaking multiplicativity: the output was not a
        # sample from any valid law, so nothing about the perturbation
        # was label-preserving (bug found by external review, reproduced
        # in tests). Ascending order makes decomp[n][0] < n available.
        for n in range(2, dim + 1):
            m, p, a = decomp[n]
            if m == 1:
                continue
            X[i, n - 1] = X[i, m - 1] * X[i, p**a - 1]
    return X
