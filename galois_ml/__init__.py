"""galois-ml: machine learning with built-in Galois structure.

Two complementary directions:

1. ML *on* number-theoretic objects (after Lee & Lee, "Machines Learn
   Number Fields, But How? The Case of Galois Groups", arXiv:2508.06670):
   classify Galois groups of Galois extensions of Q from Dedekind zeta
   coefficients, with interpretable models and theorem-backed criteria.

2. ML *with* Galois structure built in: exact GF(p^n) arithmetic, normal
   bases (in which Frobenius acts by cyclic coordinate shift), and
   Frobenius-equivariant neural architectures.
"""

from galois_ml.groups import GROUPS, TASKS, GaloisGroup
from galois_ml.zeta import euler_coefficient, zeta_coefficients
from galois_ml.data import make_dataset
from galois_ml.criteria import TheoremClassifier
from galois_ml.finite_field import GF, NormalBasis

__all__ = [
    "GROUPS",
    "TASKS",
    "GaloisGroup",
    "euler_coefficient",
    "zeta_coefficients",
    "make_dataset",
    "TheoremClassifier",
    "GF",
    "NormalBasis",
]

__version__ = "0.2.0"
