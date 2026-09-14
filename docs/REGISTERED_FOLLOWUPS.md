# Registered follow-up studies (prospective)

Written 2026-09-13, while the confirmatory study
(`confirmatory-protocol-v1`) is still executing and before any of its
results have been analyzed. These protocols are fixed here so that
their later execution cannot be read as reacting to the confirmatory
outcome. The commit containing this file is tagged
`followups-protocol-v1`. Any later change is an amendment and will be
recorded as one, with a timestamp.

None of these is part of the registered confirmatory analysis.

---

## F1. Competent-advisory baseline (`flag_routed`)

**Question.** Does an actionable warning -- naming the failed source
and the channel that remains independently verified, with no task
answers -- recover the flag-policy regression? This is the single
element the SaTML area-chair review said most changes the acceptance
verdict.

**Design.** Sextic, `combined` and `impossible` channels, theory
withheld, temperature 0.7, n = 10 per cell. Conditions: `flag`,
`flag_routed`, `repair`, `unguarded`, interleaved from a committed seed
(`INTERLEAVE_SEED_F1 = 20260913`). Same model as the confirmatory
study (Claude Sonnet 4.5) so the comparison is within-model.

**Primary contrast.** `combined`: `flag_routed` vs `flag`, raw
accuracy, one-sided exact permutation (routed higher), alpha = 0.05.

**Secondaries (Holm, family of three).** `combined` routed vs repair
(two-sided: is routed within the repair band?); `impossible` routed vs
flag; `combined` routed vs unguarded.

**Interpretation, fixed in advance.** If routed recovers to the repair
band: the pilot regression was a property of the warning design, and
the contribution sharpens to behavior-dependent mitigation vs the
conditional restoration guarantee. If routed does not recover: the
regression is not attributable to a bad warning string. Either outcome
is reportable; neither weakens the architectural claim.

## F2. Anchor factorial (mechanism of adaptive-attack robustness)

**Question.** Live agents under `data_adaptive` were not clearly harmed
(sec 11.3). Two anchors could explain this: trusted validation
feedback, and pretrained knowledge of the named groups' element orders.
Which?

**Design.** 2 x 2 under the same `data_adaptive` attack, plus clean
controls for the anonymized cells:

| | validation available | validation unavailable |
|---|---|---|
| named groups | A | B |
| opaque per-run labels | C | D |

Anonymization removes group names from EVERY exposed interface (brief,
tool outputs, guard class names), with a fresh hidden mapping per run.
The model may still recognize S3-like structure; it must not be able to
learn from pretraining which opaque label it maps to. n = 8 per cell,
interleaved (`INTERLEAVE_SEED_F2 = 20260914`). Fixed tool budgets.

**Primary endpoint.** Final class ORIENTATION (mechanically coded by
executing the returned rule, as in `export_supplement.py`), not
accuracy: fraction of runs with correct orientation, by cell.

**Prespecified predictions.** Validation-anchor hypothesis: C correct,
D degraded/inverted. Pretraining-anchor hypothesis: B correct.
Complementary: both. Clean anonymized cells failing makes the attack
comparison uninterpretable and is reported as such.

**Stress cell.** Validation corrupted too, recomputation enabled vs
disabled, to isolate the guard's causal role.

## F3. Hidden-profile test (proposed by the alignment audit)

**Question.** Can a group lose decisive, VALID private evidence through
its communication process with no agent or tool compromised? This
tests a limitation our corruption experiments cannot: repair restores
invalid evidence; it does nothing to ensure valid private evidence is
shared or used.

**Design.** Three agents, one communication round, one designated
decider. Two agents receive shared evidence only; one additionally
receives a trusted tool output containing a decisive coefficient.
Conditions: HIDDEN (only one agent has it) vs COMMON (all have it).
Cases: CONFLICT (private evidence reverses the shared provisional
answer) vs CONGRUENT (confirms it). Rotate the informed agent;
counterbalance optimal action and report order. Agents are told some
information may be private.

**Measured, separately, along the information path.** (1) Did the
informed agent compute the correct action? (2) Did its report transmit
the coefficient accurately? (3) Was it available to the decider? (4)
Did the decider use it? (5) Final regret.

**Primary contrast.** Increase in error from COMMON to HIDDEN on
CONFLICT cases, CONGRUENT as control; independent group runs are the
units. A few dozen matched blocks as a pilot.

**Architecture arm (optional, separate claim).** Pass final candidates
through the release gate as usual; report acceptance, fallback, and
released risk. This tests protection against the downstream
consequence, NOT restoration of information pooling, and is reported
as such.

## F4. Second model family

After F1, rerun the confirmatory primary contrast (`combined`, repair
vs flag) and F1's primary with a non-Anthropic model family available
on Bedrock. Reported per model; no pooling across families.

---

## Not registered here

The confirmatory study's own secondaries are registered in
`experiments/confirmatory_eval.py` (tag `confirmatory-protocol-v1`).
The flag-vs-unguarded contrast ("warnings harmed agents") is NOT
registered anywhere and will be reported as exploratory.
