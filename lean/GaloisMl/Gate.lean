/-!
# The release gate: algebraic guarantee given the concentration event

The probabilistic content of Theorem 1 is a single event

    E := ∀ j, L(h_j) - L(h_0) ≤ Δ̂_j + r_j

which Hoeffding / empirical Bernstein give with probability ≥ 1 - δ.
Everything else in the theorem is deterministic arithmetic, and that is
what this file proves: ON the event `E`, whatever the release rule
outputs has risk at most `L(h_0) + ε`. The proof of `E` itself needs
measure theory (Mathlib) and is not formalized here; the split makes
explicit that the gate's correctness reduces to one standard
inequality plus the arithmetic below.

Quantities are integers in a common unit (e.g. multiples of `1/m`, or
any fixed grid): every risk, empirical difference, margin and ε in the
implementation is a rational with a common denominator, so integer
arithmetic on the numerators is fully general and lets `omega` close
the goals without Mathlib.
-/

namespace GaloisMl.Gate

inductive Released (n : Nat)
  | fallback
  | cand (j : Fin n)

structure Instance (n : Nat) where
  riskFallback : Int
  risk   : Fin n → Int   -- population risk L(h_j)
  dhat   : Fin n → Int   -- empirical paired loss difference Δ̂_j
  margin : Fin n → Int   -- r_j
  eps    : Int

/-- The concentration event: every candidate's true excess risk is
bounded by its empirical excess plus its margin. -/
def ConcEvent {n} (I : Instance n) : Prop :=
  ∀ j, I.risk j - I.riskFallback ≤ I.dhat j + I.margin j

/-- A candidate is releasable iff its certified bound is within ε. -/
def Releasable {n} (I : Instance n) (j : Fin n) : Prop :=
  I.dhat j + I.margin j ≤ I.eps

def riskOf {n} (I : Instance n) : Released n → Int
  | .fallback => I.riskFallback
  | .cand j   => I.risk j

/-- A release is valid iff it is the fallback or a releasable candidate. -/
def ValidRelease {n} (I : Instance n) : Released n → Prop
  | .fallback => True
  | .cand j   => Releasable I j

/-- THEOREM (deterministic clause of the gate). On the concentration
event, every valid release has risk ≤ L(h_0) + ε. -/
theorem released_risk_le {n} (I : Instance n) (hε : 0 ≤ I.eps)
    (hE : ConcEvent I) (out : Released n) (hv : ValidRelease I out) :
    riskOf I out ≤ I.riskFallback + I.eps := by
  cases out with
  | fallback =>
    simp only [riskOf]
    omega
  | cand j =>
    simp only [riskOf]
    have h1 := hE j
    have h2 : Releasable I j := hv
    unfold Releasable at h2
    omega

/-- Why `ε ≥ 0` is a hypothesis and not a nicety (review sec 3.4): with
ε < 0 even the fallback violates the bound, so the claimed guarantee
fails with probability 1. -/
theorem fallback_violates_of_neg_eps {n} (I : Instance n) (hε : I.eps < 0) :
    ¬ (riskOf I (Released.fallback (n := n)) ≤ I.riskFallback + I.eps) := by
  simp only [riskOf]
  omega

/-- Post-hoc selection among accepted candidates is covered: the theorem
quantifies over ANY valid release, which is exactly why the event must
hold for all `j` simultaneously (the K-fold union bound in the margin). -/
theorem any_selection_ok {n} (I : Instance n) (hε : 0 ≤ I.eps)
    (hE : ConcEvent I) (j : Fin n) (hj : Releasable I j) :
    I.risk j ≤ I.riskFallback + I.eps :=
  released_risk_le I hε hE (.cand j) hj

/-- Fail-closed is safe: releasing the fallback never violates the
bound, regardless of the event. This is why evaluation failures may
always default to the fallback. -/
theorem fallback_always_safe {n} (I : Instance n) (hε : 0 ≤ I.eps) :
    riskOf I (Released.fallback (n := n)) ≤ I.riskFallback + I.eps := by
  simp only [riskOf]
  omega

end GaloisMl.Gate
