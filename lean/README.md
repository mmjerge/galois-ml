# Machine-checked statements (Lean 4, Mathlib-free)

```bash
cd lean && lake build GaloisMl     # ~10 s; Lean 4.33, no dependencies
```

What is formalized, and the exact boundary of what is not:

| file | statements | corresponds to |
|---|---|---|
| `Attainable.lean` | `attainable d k` from Pascal-rule binomials; `zero_attainable_iff` (0 is attainable iff some divisor of `d` fails to divide `k`); concrete instances `attainable_6_6 = [462,10,3,1]`, `zero_not_attainable_6_6`, `attainable_4_2`, `attainable_6_1/6_2` | the `theorem` guarantee class in `guard.py`; the zero-inclusion bug the first external review found is now a theorem it cannot recur against |
| `Gate.lean` | `released_risk_le`: on the concentration event, every valid release has risk ≤ L(h₀)+ε; `fallback_violates_of_neg_eps`; `any_selection_ok`; `fallback_always_safe` | the deterministic clause of Theorem 1 (paper) / `certify_release` |
| `Indistinguishability.lean` | `relabeling_undetected`: closure under π + soundness ⇒ relabeled honest transcripts are accepted; `detection_needs_anchor_or_fp` (contrapositive) | Proposition 1, zero-false-positive form |
| `Repair.lean` | `clean_transcript_equivalence`: exact repair + coupled agent policy ⇒ identical histories at every step; `artifact_equivalence`; `exact_repair_iff` | Proposition 2 |

**Not formalized, deliberately:** the probabilistic inequality behind
the gate (Hoeffding / empirical Bernstein). `Gate.lean` takes the
concentration event as a hypothesis and proves everything downstream of
it, so the gate's correctness is reduced to one standard inequality plus
machine-checked arithmetic. Formalizing the inequality itself requires
Mathlib's measure theory; that is a separate, larger project. Likewise
the α-false-positive version of Proposition 1 needs probability and is
stated only on paper.

Quantities in `Gate.lean` are integers in a common unit; every risk,
margin, and ε in the implementation is a rational with a common
denominator, so this is fully general.
