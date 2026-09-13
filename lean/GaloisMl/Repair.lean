/-!
# Clean-transcript equivalence under exact repair (Proposition 2)

A pipeline is a deterministic transition system once agent randomness is
fixed (coupled): from a history `h` the agent issues a query `q h`; the
environment answers; the history extends. Two executions differ only in
the response channel:

* clean:    response = `honest q`
* attacked: response = `mediate (attack q)`

where `mediate` is the guard with repair. The hypothesis of exact
repair is that `mediate (attack q) = honest q` for every query. The
conclusion is that the two histories coincide at every step, hence
every derived artifact coincides.

The proof is the induction the paper sketches. Formalizing it pins the
hypotheses: determinism of the agent given history (coupled
randomness), and pointwise equality of mediated and honest responses.
Timing, resource, and side channels are outside the model by
construction of `step` -- which is exactly the scope caveat in the
paper.
-/

namespace GaloisMl.Repair

variable {History Query Response : Type}

structure Pipeline (History Query Response : Type) where
  query  : History → Query                      -- agent policy (coupled randomness)
  extend : History → Query → Response → History  -- append an exchange

def step (P : Pipeline History Query Response)
    (respond : Query → Response) (h : History) : History :=
  P.extend h (P.query h) (respond (P.query h))

def run (P : Pipeline History Query Response)
    (respond : Query → Response) (h0 : History) : Nat → History
  | 0     => h0
  | n + 1 => step P respond (run P respond h0 n)

/-- Exact repair: the mediated attacked response equals the honest one
for every query. -/
def ExactRepair (honest attack : Query → Response)
    (mediate : Response → Response) : Prop :=
  ∀ q, mediate (attack q) = honest q

/-- PROPOSITION. Under exact repair and coupled agent randomness, the
attacked-and-mediated execution and the clean execution have identical
histories at every step. -/
theorem clean_transcript_equivalence
    (P : Pipeline History Query Response)
    (honest attack : Query → Response) (mediate : Response → Response)
    (hrep : ExactRepair honest attack mediate) (h0 : History) :
    ∀ n, run P (fun q => mediate (attack q)) h0 n = run P honest h0 n := by
  intro n
  induction n with
  | zero => rfl
  | succ n ih =>
    simp only [run, step]
    rw [ih, hrep]

/-- Corollary: any artifact that is a function of the final history is
identical, so attack-induced degradation is exactly zero. -/
theorem artifact_equivalence {Artifact : Type}
    (P : Pipeline History Query Response)
    (honest attack : Query → Response) (mediate : Response → Response)
    (hrep : ExactRepair honest attack mediate) (h0 : History)
    (artifact : History → Artifact) (n : Nat) :
    artifact (run P (fun q => mediate (attack q)) h0 n)
      = artifact (run P honest h0 n) := by
  rw [clean_transcript_equivalence P honest attack mediate hrep h0 n]

/-- What repair does NOT give: if mediation is only identity on
undetected responses (the `data` channel), the hypothesis fails at any
query the attack corrupts undetectably, and the induction does not go
through. Stated as: exact repair is equivalent to the attacked channel
being pointwise honest after mediation -- nothing weaker suffices for
this proof. -/
theorem exact_repair_iff
    (honest attack : Query → Response) (mediate : Response → Response) :
    ExactRepair honest attack mediate ↔
      (fun q => mediate (attack q)) = honest := by
  constructor
  · intro h; funext q; exact h q
  · intro h q; exact congrFun h q

end GaloisMl.Repair
