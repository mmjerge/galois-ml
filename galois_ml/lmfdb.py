"""Real number fields from the LMFDB, and zeta coefficients computed
from defining polynomials.

For a Galois extension K = Q[x]/(F) and a prime p not dividing
[O_K : Z[x]/(F)], the factorization of F mod p mirrors the factorization
of (p) in O_K (Dedekind's theorem): the irreducible factors all have the
same degree f (the inertia degree) and the number of *distinct* factors
is g. The Euler factor (1 - p^{-fs})^{-g} then determines every a_{p^k},
and multiplicativity gives all a_n - for ramified primes too, since
Proposition 2.1 of Lee & Lee shows e does not enter the Euler factor.

Caveat: at the finitely many primes dividing the index [O_K : Z[alpha]]
the factorization mod p can misreport the decomposition type. LMFDB's
reduced (polredabs) polynomials keep this rare, and it perturbs at most
a few features per field.
"""

from __future__ import annotations

import json
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
from sympy import GF as _GF
from sympy import Poly, symbols

from galois_ml.groups import GROUPS, TASKS
from galois_ml.zeta import primes_up_to, zeta_coefficients

API_URL = "https://www.lmfdb.org/api/nf_fields/"

#: LMFDB nTt Galois labels for the groups in this project.
GALOIS_LABELS: dict[str, str] = {
    "C4": "4T1", "C2^2": "4T2",
    "C6": "6T1", "S3": "6T2",
    "C8": "8T1", "C4xC2": "8T2", "C2^3": "8T3",
    "D4": "8T4", "Q8": "8T5",
    "C9": "9T1", "C3^2": "9T2",
    "C10": "10T1", "D5": "10T2",
}


def fetch_fields(
    group_name: str,
    limit: int = 200,
    cache_dir: str | Path = ".lmfdb_cache",
) -> list[dict]:
    """Fetch defining polynomials of Galois fields with the given group
    from the LMFDB API (cached on disk as JSON).

    Returns a list of {"label": str, "coeffs": [c0, ..., cn]} with
    ascending coefficients, cn = 1 (monic).
    """
    label = GALOIS_LABELS[group_name]
    cache = Path(cache_dir) / f"{label}_{limit}.json"
    if cache.exists():
        return json.loads(cache.read_text())

    degree = GROUPS[group_name].order
    records: list[dict] = []
    offset = 0
    while len(records) < limit:
        query = urllib.parse.urlencode({
            "degree": degree,
            "galois_label": label,
            "_format": "json",
            "_fields": "label,coeffs",
            "_offset": offset,
        })
        with urllib.request.urlopen(f"{API_URL}?{query}", timeout=60) as r:
            payload = json.load(r)
        batch = payload.get("data", [])
        if not batch:
            break
        records.extend(
            {"label": rec["label"], "coeffs": rec["coeffs"]} for rec in batch
        )
        offset += len(batch)
    records = records[:limit]

    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(records))
    return records


def decomposition_from_polynomial(
    coeffs: list[int], N: int
) -> dict[int, tuple[int, int]]:
    """Decomposition data (f_p, g_p) for each prime p <= N, by factoring
    the monic defining polynomial mod p (Dedekind's theorem). For a
    Galois field every irreducible factor has the same degree f, the
    number of distinct factors is g, and repeated factors signal
    ramification (e > 1) - which does not affect the Euler factor."""
    x = symbols("x")
    poly = Poly(list(reversed(coeffs)), x)  # sympy wants descending
    degree = poly.degree()
    decomposition: dict[int, tuple[int, int]] = {}
    for p in primes_up_to(N):
        factors = poly.set_domain(_GF(p)).factor_list()[1]
        distinct_degrees = {f.degree() for f, _ in factors}
        if len(distinct_degrees) == 1:
            f = distinct_degrees.pop()
            g = len(factors)
        else:
            # Only possible at index-dividing primes, where Dedekind's
            # correspondence fails. Fall back to an unramified guess.
            f = max(distinct_degrees)
            g = degree // f
        if degree % f != 0 or f * g > degree or degree % (f * g) != 0:
            f, g = 1, degree  # defensive; should not occur for Galois K
        decomposition[p] = (f, g)
    return decomposition


def make_lmfdb_dataset(
    task: str,
    per_class: int = 100,
    n_coeffs: int = 1000,
    cache_dir: str | Path = ".lmfdb_cache",
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Zeta-coefficient dataset from real LMFDB fields, with the same
    format and label convention as `galois_ml.data.make_dataset`."""
    class_names = TASKS[task]
    rows, labels = [], []
    for gi, name in enumerate(class_names):
        for rec in fetch_fields(name, limit=per_class, cache_dir=cache_dir):
            dec = decomposition_from_polynomial(rec["coeffs"], n_coeffs)
            rows.append(zeta_coefficients(n_coeffs, dec, GROUPS[name].order))
            labels.append(gi)
    return np.array(rows), np.array(labels), list(class_names)
