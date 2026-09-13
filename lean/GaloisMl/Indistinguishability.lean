/-!
# Restricted label-permutation indistinguishability (zero-false-positive form)

Proposition 1 of the paper, in its deterministic form. A verifier `V`
maps transcripts to accept/reject. `Honest` is the set of transcripts an
honest world can produce; `π` is a relabeling action on transcripts.

Hypotheses:
* `closed`: the honest set is closed under `π` (the verifier's trusted
  knowledge is label-permutation-invariant, so a relabeled honest world
  is another admissible honest world);
* `sound`: `V` accepts every honest transcript (zero false positives).

Conclusion: `V` accepts every relabeled honest transcript, i.e. the
consistent-relabeling attack is never detected.

This is deliberately the trivial-looking core. Its value is in what it
forces to be explicit: the ONLY hypotheses are closure and soundness.
A verifier with class-anchored knowledge breaks `closed`; a verifier
with false positives breaks `sound` (and the paper's α-version bounds
detection by the false-positive rate, which needs probability and is
not formalized here).
-/

namespace GaloisMl.Indist

variable {Transcript : Type}

structure Setting (Transcript : Type) where
  Honest : Transcript → Prop
  π      : Transcript → Transcript
  V      : Transcript → Bool      -- true = accept

def Closed (S : Setting Transcript) : Prop :=
  ∀ t, S.Honest t → S.Honest (S.π t)

def Sound (S : Setting Transcript) : Prop :=
  ∀ t, S.Honest t → S.V t = true

/-- PROPOSITION. Under closure and soundness, the relabeled transcript
of any honest world is accepted: detection probability is zero. -/
theorem relabeling_undetected (S : Setting Transcript)
    (hc : Closed S) (hs : Sound S) (t : Transcript) (ht : S.Honest t) :
    S.V (S.π t) = true :=
  hs (S.π t) (hc t ht)

/-- The contrapositive makes the escape routes explicit: if some
relabeled honest transcript IS rejected, the verifier cannot be both
closed-under-π and sound -- it has class-anchored knowledge, or a false
positive somewhere. -/
theorem detection_needs_anchor_or_fp (S : Setting Transcript)
    (t : Transcript) (ht : S.Honest t) (hrej : S.V (S.π t) = false) :
    ¬ (Closed S ∧ Sound S) := by
  rintro ⟨hc, hs⟩
  have := relabeling_undetected S hc hs t ht
  rw [this] at hrej
  exact Bool.noConfusion hrej

end GaloisMl.Indist
