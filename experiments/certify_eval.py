"""Certified audit of standard ML models against a per-input ground-truth
posterior. No model API, no credentials, fully offline.

Four audits that are normally done against proxies, done here exactly:

1. CALIBRATION. Excess log loss = KL(true || model) in bits. Zero iff the
   model reproduces the law. The true posterior's entropy is the
   irreducible floor, so everything above it is attributable to the
   model rather than to task difficulty. No binning is involved.
2. ADMISSIBILITY. Decidable rejection of inputs the generative law
   forbids, with soundness by construction (honest data is never
   rejected). Compared against the usual heuristic: low model
   confidence.
3. ATTRIBUTION. Feature importances scored against exact per-feature
   mutual information; features with MI = 0 provably carry no signal.
4. INVARIANCE. Perturbations that provably preserve the true label
   (resampling a prime's signature from the same class), so a flipped
   prediction is a certified defect. Flips are only counted where the
   true posterior still favours the original class, which the exact
   posterior lets us check rather than assume.

Run:  python experiments/certify_eval.py [--tasks sextic quartic] [--seeds 3]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parent.parent))

from galois_ml.benchmark import load_benchmark
from galois_ml.certify import (
    admissible_mask,
    calibration_report,
    certified_perturbation,
    exact_posterior,
)
from galois_ml.groups import GROUPS

OUT = Path("experiments/outputs")
N_COEFFS = 200  # exact posterior is per-row; keep the audit quick


def build_models(seed: int):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.tree import DecisionTreeClassifier

    # The linear and neural models are scaled: unscaled, LogisticRegression
    # does not converge on raw coefficient magnitudes, and a convergence
    # failure would confound every audit below with an optimization
    # artifact. Trees are scale-invariant and left alone.
    return {
        "decision_tree": DecisionTreeClassifier(random_state=seed),
        "random_forest": RandomForestClassifier(
            n_estimators=200, random_state=seed
        ),
        "logistic_regression": make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=5000)
        ),
        "mlp": make_pipeline(
            StandardScaler(),
            MLPClassifier(hidden_layer_sizes=(64, 64), max_iter=1500,
                          random_state=seed),
        ),
    }


def audit(bench, name, model, seed: int) -> dict:
    degree = GROUPS[bench.class_names[0]].order
    model.fit(bench.X_train, bench.y_train)

    post = exact_posterior(bench.task, bench.X_test)
    probs = model.predict_proba(bench.X_test)
    rep = calibration_report(probs, post, bench.y_test)
    row = {"task": bench.task, "model": name, "seed": seed,
           "n_coeffs": bench.X_train.shape[1], **rep.to_dict()}

    # -- attribution against exact MI ---------------------------------
    est = model[-1] if hasattr(model, "steps") else model  # unwrap pipeline
    imp = None
    if hasattr(est, "feature_importances_"):
        imp = np.asarray(est.feature_importances_, dtype=float)
    elif hasattr(est, "coef_"):
        imp = np.abs(np.asarray(est.coef_, dtype=float)).sum(axis=0)
    if imp is not None:
        row["attribution"] = bench.feature_recovery(imp, top_k=20)
        # provable spurious reliance: mass placed on MI = 0 features
        zero_mi = bench.feature_mi <= 1e-12
        total = float(imp.sum())
        row["mass_on_zero_information_features"] = (
            float(imp[zero_mi].sum() / total) if total > 0 else None
        )

    # -- admissibility: decidable rejection vs confidence heuristic ----
    rng = np.random.default_rng(seed)
    X_bad = bench.X_test.copy()
    # corrupt one prime-power coefficient per row to a forbidden value,
    # the input-level analogue of the guard's `impossible` channel
    cols = rng.choice([3, 8, 24, 48], size=len(X_bad))
    for i, c in enumerate(cols):
        X_bad[i, c] = 7.0
    mask_good = admissible_mask(bench.X_test, degree)
    mask_bad = admissible_mask(X_bad, degree)
    conf_good = probs.max(axis=1)
    conf_bad = model.predict_proba(X_bad).max(axis=1)
    thresh = float(np.quantile(conf_good, 0.05))  # 5% false-positive budget
    row["admissibility"] = {
        "false_rejections_on_honest_data": int((~mask_good).sum()),
        "detected_corrupt_rows": int((~mask_bad).sum()),
        "n_corrupt": int(len(X_bad)),
        "detection_rate": float((~mask_bad).mean()),
        # the usual heuristic, for comparison, given a 5% FP budget
        "confidence_heuristic_detection_rate": float((conf_bad < thresh).mean()),
        "confidence_heuristic_false_positive_rate": float(
            (conf_good < thresh).mean()
        ),
        # does corruption even move the model? (it should not be trusted
        # to react: the values are impossible, not merely unusual)
        "mean_confidence_honest": float(conf_good.mean()),
        "mean_confidence_corrupt": float(conf_bad.mean()),
    }

    # -- invariance under provably label-preserving perturbation -------
    X_pert = certified_perturbation(
        bench.X_test, bench.task, class_index=bench.y_test,
        n_coeffs=bench.X_train.shape[1], seed=seed,
    )
    post_pert = exact_posterior(bench.task, X_pert)
    pred_before = model.predict(bench.X_test)
    pred_after = model.predict(X_pert)
    # Bayes-label-stable resampling degradation (tightened on external
    # review): a flip only counts as a defect when (a) the same class
    # wins the true posterior at both endpoints AND (b) the original
    # prediction agreed with it. Without (b), a flip can be a
    # correction; without (a), the evidence genuinely moved.
    truth_stable = np.argmax(post_pert, axis=1) == np.argmax(post, axis=1)
    initially_right = pred_before == np.argmax(post, axis=1)
    eligible = truth_stable & initially_right
    flipped = (pred_before != pred_after) & eligible
    corrections = ((pred_before != pred_after) & truth_stable
                   & ~initially_right
                   & (pred_after == np.argmax(post_pert, axis=1)))
    row["invariance"] = {
        "n_truth_stable": int(truth_stable.sum()),
        "n_eligible": int(eligible.sum()),
        "certified_flips": int(flipped.sum()),
        "certified_flip_rate": (float(flipped.sum() / eligible.sum())
                                if eligible.sum() else None),
        "corrective_flips": int(corrections.sum()),
        "accuracy_before": float(np.mean(pred_before == bench.y_test)),
        "accuracy_after": float(np.mean(pred_after == bench.y_test)),
    }
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="+",
                    default=["quartic", "sextic", "degree21"])
    ap.add_argument("--n-coeffs", type=int, default=N_COEFFS)
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args()

    rows = []
    for task in args.tasks:
        bench = load_benchmark(task, n_coeffs=args.n_coeffs)
        print(f"\n== {task} (ceiling {bench.bayes_accuracy:.4f}, "
              f"{args.n_coeffs} coeffs) ==")
        print(f"{'model':<22}{'acc':>7}{'regret':>9}{'excess bits':>13}"
              f"{'ECE-exact':>11}{'overconf':>10}{'flips':>8}{'MI=0 mass':>11}")
        # 'flips' = certified invariance defects: prediction changed under a
        # provably label-preserving perturbation, counted only where the
        # true posterior still favours the original class
        for seed in range(args.seeds):
            for name, model in build_models(seed).items():
                r = audit(bench, name, model, seed)
                rows.append(r)
                if seed == 0:
                    zm = r.get("mass_on_zero_information_features")
                    print(f"{name:<22}{r['accuracy']:>7.4f}"
                          f"{r['accuracy_regret']:>+9.4f}"
                          f"{r['excess_log_loss']:>13.4f}"
                          f"{r['exact_calibration_error']:>11.4f}"
                          f"{r['overconfidence']:>+10.4f}"
                          f"{(r['invariance']['certified_flip_rate'] or 0):>8.1%}"
                          + (f"{zm:>11.3f}" if zm is not None else f"{'-':>11}"))
        adm = rows[-1]["admissibility"]
        print(f"  admissibility (decidable): {adm['detection_rate']:.0%} of "
              f"corrupt inputs rejected, "
              f"{adm['false_rejections_on_honest_data']} false rejections")
        print(f"  confidence heuristic @5% FP budget: "
              f"{adm['confidence_heuristic_detection_rate']:.0%} detected, "
              f"FP {adm['confidence_heuristic_false_positive_rate']:.0%}")
        print(f"  model confidence on impossible inputs: "
              f"{adm['mean_confidence_corrupt']:.3f} vs "
              f"{adm['mean_confidence_honest']:.3f} on honest")

    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "certify_audit.json"
    path.write_text(json.dumps(rows, indent=1))
    print(f"\nresults -> {path}")


if __name__ == "__main__":
    main()
