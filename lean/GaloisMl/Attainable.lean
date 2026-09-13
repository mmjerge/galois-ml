/-!
# Attainable Euler-factor coefficients (unramified model)

For a Galois field of degree `d` and an unramified prime `p` with residue
degree `f ∣ d` and `g = d / f` primes above it,

    a_{p^k} = C(g + k/f - 1, g - 1)   if f ∣ k,   else 0.

`attainable d k` is the list of values over all divisors `f` of `d`. The
guard in `galois_ml/guard.py` checks reported values against exactly
this set. The theorem `zero_attainable_iff` is the fact an external
review found the implementation violating (it seeded `{0}`
unconditionally): zero is attainable iff SOME divisor of `d` fails to
divide `k`. `attainable_6_6` pins the counterexample.

Mathlib-free: binomials are defined by Pascal's rule so that every
concrete instance is decidable by evaluation.
-/

namespace GaloisMl

/-- Binomial coefficient by Pascal's rule. -/
def binom : Nat → Nat → Nat
  | _, 0 => 1
  | 0, _ + 1 => 0
  | n + 1, k + 1 => binom n k + binom n (k + 1)

theorem binom_zero_right (n : Nat) : binom n 0 = 1 := by
  cases n <;> rfl

/-- `C(n, k) > 0` whenever `k ≤ n`. -/
theorem binom_pos : ∀ {n k : Nat}, k ≤ n → 0 < binom n k
  | n, 0, _ => by rw [binom_zero_right]; exact Nat.zero_lt_one
  | 0, k + 1, h => by omega
  | n + 1, k + 1, h => by
    have hk : k ≤ n := Nat.le_of_succ_le_succ h
    have := binom_pos hk
    simp only [binom]
    omega

/-- Euler-factor coefficient of `p^k` for residue degree `f`, with `g`
primes above `p`. -/
def eulerCoeff (k f g : Nat) : Nat :=
  if k % f = 0 then binom (g + k / f - 1) (g - 1) else 0

/-- Positive divisors of `d`. -/
def divisors (d : Nat) : List Nat :=
  (List.range (d + 1)).filter (fun f => f != 0 && d % f == 0)

/-- All values `a_{p^k}` can take for any Galois field of degree `d` at
an unramified prime, over the unramified decomposition types. -/
def attainable (d k : Nat) : List Nat :=
  (divisors d).map (fun f => eulerCoeff k f (d / f))

theorem mem_divisors {d f : Nat} :
    f ∈ divisors d ↔ f ≠ 0 ∧ d % f = 0 ∧ f ≤ d := by
  unfold divisors
  rw [List.mem_filter, List.mem_range]
  simp only [Bool.and_eq_true, bne_iff_ne, ne_eq, beq_iff_eq]
  constructor
  · rintro ⟨hlt, hne, hmod⟩
    exact ⟨hne, hmod, Nat.lt_succ_iff.mp hlt⟩
  · rintro ⟨hne, hmod, hle⟩
    exact ⟨Nat.lt_succ_iff.mpr hle, hne, hmod⟩

/-- When `f ∣ k` and `g ≥ 1`, the Euler coefficient is positive. -/
theorem eulerCoeff_pos {k f g : Nat} (_hg : 1 ≤ g) (hdiv : k % f = 0) :
    0 < eulerCoeff k f g := by
  unfold eulerCoeff
  rw [if_pos hdiv]
  apply binom_pos
  exact Nat.sub_le_sub_right (Nat.le_add_right g (k / f)) 1

/-- When `f ∤ k`, the Euler coefficient is zero. -/
theorem eulerCoeff_eq_zero {k f g : Nat} (hndiv : k % f ≠ 0) :
    eulerCoeff k f g = 0 := by
  unfold eulerCoeff
  rw [if_neg hndiv]

/-- THE ZERO-INCLUSION THEOREM. Zero is attainable for `a_{p^k}` in
degree `d` iff some divisor `f` of `d` does not divide `k`. In
particular, when every divisor divides `k` (e.g. `d ∣ k`), a reported
zero is impossible. -/
theorem zero_attainable_iff {d k : Nat} :
    0 ∈ attainable d k ↔ ∃ f, f ∈ divisors d ∧ k % f ≠ 0 := by
  unfold attainable
  rw [List.mem_map]
  constructor
  · rintro ⟨f, hf, hzero⟩
    refine ⟨f, hf, fun hdiv => ?_⟩
    have hf' := mem_divisors.mp hf
    have hg : 1 ≤ d / f := Nat.div_pos hf'.2.2 (Nat.pos_of_ne_zero hf'.1)
    have := eulerCoeff_pos (k := k) (f := f) hg hdiv
    omega
  · rintro ⟨f, hf, hndiv⟩
    exact ⟨f, hf, eulerCoeff_eq_zero hndiv⟩

/-- Corollary in the form the guard needs: if EVERY divisor of `d`
divides `k`, then zero is not attainable. -/
theorem zero_not_attainable_of_all_divide {d k : Nat}
    (h : ∀ f, f ∈ divisors d → k % f = 0) : 0 ∉ attainable d k := by
  rw [zero_attainable_iff]
  rintro ⟨f, hf, hnd⟩
  exact hnd (h f hf)

/-- The review's counterexample: degree 6, k = 6. Every divisor of 6
divides 6, so zero is NOT attainable; the exact set is {462, 10, 3, 1}
for f = 1, 2, 3, 6. -/
theorem attainable_6_6 : attainable 6 6 = [462, 10, 3, 1] := by decide

theorem zero_not_attainable_6_6 : 0 ∉ attainable 6 6 := by decide

/-- Sanity: degree 4, k = 2 gives {10, 2, 0} for f = 1, 2, 4 -- matching
the Python reference `attainable_coefficient_values(4, 2) == {0, 10, 2}`. -/
theorem attainable_4_2 : attainable 4 2 = [10, 2, 0] := by decide

/-- Degree 6, k = 2: f = 1 → C(7,5) = 21, f = 2 → C(3,2) = 3, f = 3, 6 → 0.
This is the same-prime consistency fact used in the paper's example
(a_2 = 6 forces f = 1, hence a_4 = 21). -/
theorem attainable_6_2 : attainable 6 2 = [21, 3, 0, 0] := by decide
theorem attainable_6_1 : attainable 6 1 = [6, 0, 0, 0] := by decide

end GaloisMl
