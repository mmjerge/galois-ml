"""Reproduce the ML experiments of Lee & Lee (arXiv:2508.06670) on
synthetic Chebotarev data: decision trees and logistic regression
classify Galois groups from zeta coefficients, compared against the
theorem-backed classifier derived from the paper's proofs.

Run:  python experiments/classify_galois_groups.py
"""

from galois_ml.baselines import run_all


def main() -> None:
    results = run_all(n_samples=3000, n_coeffs=1000, seed=0)

    header = (
        f"{'task':<18} {'classes':<22} {'tree':>7} {'logreg':>7} {'theorem':>8}"
    )
    print(header)
    print("-" * len(header))
    for r in results:
        print(
            f"{r.task:<18} {'/'.join(r.class_names):<22} "
            f"{r.accuracy['decision_tree']:>7.4f} "
            f"{r.accuracy['logistic_regression']:>7.4f} "
            f"{r.accuracy['theorem']:>8.4f}"
        )

    print("\nTop decision-tree features (what the model looks at):")
    for r in results:
        print(f"\n  {r.task} ({' vs '.join(r.class_names)}):")
        for feat in r.top_tree_features[:5]:
            print(f"    {feat}")


if __name__ == "__main__":
    main()
