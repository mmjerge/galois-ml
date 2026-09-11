# Positioning: where this testbed sits in agentic AI security

This note records the specific correspondences between this repository's
experiments and two external bodies of work, with an explicit separation
between claims that are directly supported by those sources and
inferences that are our own. It exists so the framing can be checked
rather than taken on faith.

## 1. NDS2 Lab / Agentic AI Security Initiative (Northeastern)

Verified from the lab's project index (agentaisec.github.io, last
updated 2026-06-14). Mapping of our components to their published lines:

| This repo | Their work | Their stated scope |
|---|---|---|
| `injection` tool-poisoning channel (`galois_ml/tool_poisoning.py`) | MUZZLE (arXiv:2602.09222), ACE (arXiv:2504.20984) | Web agents "misled, hijacked, or otherwise driven toward unintended behavior via indirect prompt injection"; defenses against indirect prompt injection |
| `saboteur` / `saboteur_critic` conditions (`galois_ml/multi_agent.py`) | Google Research Award 2025; SAGA-BFT (arXiv:2605.12364) | "Secure distributed architectures that prevent compromised agents from undermining task execution" |
| Critic verification, detection rate, recovery measurement | ACE (arXiv:2504.20984) | "Mechanisms for detecting and recovering from compromised actions in multi-step agentic pipelines" |
| Analyst/critic discovery pipeline (Strands on Bedrock) | APWA (arXiv:2605.15132) | "Architectures that enable multiple agents to securely interact and collaborate in solving complex tasks" |
| Poisoning with recomputable Bayes ceiling (`galois_ml/benchmark.py`, `tool_poisoning.py`) | SleeperNets (NeurIPS 2024), DAZE (arXiv:2602.05089), Q-Incept (arXiv:2410.13995) | "Training-time backdoor and poisoning threats" (their setting is RL; ours is agentic/supervised) |
| Correlated-failure analysis (`galois_ml/conformity.py`) | SAGA-BFT | Byzantine-style resilience arguments depend on independence assumptions, which this measures directly |

**What this testbed adds to that agenda.** Their work builds defenses and
architectures; the recurring evaluation difficulty is quantifying *how
much* harm an attack causes or a defense prevents, because in realistic
agentic tasks the correct outcome is not known in closed form. Here it
is: every task ships a provable Bayes-optimal ceiling, exact per-feature
mutual information, and a reference theorem. Harm becomes exact regret,
defense value becomes exact recovered regret, and leakage becomes a bit
quantity rather than an estimate.

**What it does not cover.** This is an *epistemic* security testbed:
correctness, faithfulness, provenance, and robustness of conclusions. It
says nothing about sandbox escape, credential theft, privilege
escalation, or tool-use permission boundaries, which is where SAGA and
MUZZLE operate. The two are complementary, not substitutes.

## 2. Anthropic Frontier Red Team, "Patterns and problems in emerging
multiagent systems" (2026-08-13)

### Directly supported by that report

- **Regret-against-ceiling is their methodology too.** Their
  lie-detection experiment scores models "against a naive policy that
  trusts every report, and against an oracle with perfect discovery,"
  reporting how much of that gap a model recovers. This is the same
  measurement structure used throughout this repo. Difference in kind:
  their ground truth is stipulated by scenario construction (a scripted
  scout lies at a fixed rate), whereas ours is derived from a theorem,
  so the ceiling is computable rather than assumed and instances can be
  generated without limit.
- **Detectable-in-principle, missed-in-practice.** In their setup, "the
  overlap in their reports makes it possible for the listener to detect
  lies in principle, since a false report will eventually contradict an
  honest one," yet models underperform the oracle. Our `impossible`
  poisoning channel is the mathematical analogue: it reports
  coefficient values the Euler-factor formula forbids for any
  decomposition type, so a lie is provable from a single tool output
  given facts the agent already holds.
- **The gap is disposition, not knowledge.** Their conclusion: "Every
  model we tested abstractly understands that information sources have
  their own incentives, and that consensus is not necessarily evidence.
  What is missing is a disposition to act on that knowledge without
  prompting." This is the behaviour our detection-rate metric targets.
- **Agents are low variance and fail together.** "When one agent makes a
  bad decision, it is likely that many agents will make that same bad
  decision. What would have been isolated problems can quickly become
  systemic failures," with concrete instances (18 of 30 agents choosing
  the identical branch name; independent agents producing identically
  titled stories).

### Our inference, not their claim

- That agent conformity undermines the independence premise behind
  voting, review, and Byzantine-tolerance arguments. Their report
  documents the conformity; the implication for redundancy-based
  assurance is our extrapolation, and `galois_ml/conformity.py` exists
  to test it rather than to assert it.

### Our own preliminary results, stated at their actual strength

- A single-seed tool-poisoning run showed 0% attack detection across all
  eight conditions, and critic verdicts of "accept" in nearly every
  case. This is *consistent with* their gullibility finding; it is not a
  replication. n = 8, one seed, one task, and detection was measured by
  a deliberately generous keyword heuristic (which over-counts
  suspicion, making a low rate conservative but still coarse).
- That run also had a design flaw worth recording: the task brief listed
  the candidate groups' element orders, an honest un-poisonable channel
  sufficient to derive the criterion without tools, so the agent reached
  the correct rule while reading falsified data. The attack had no path
  to succeed. Measuring channel harm requires withholding that
  redundancy (`--no-theory-in-brief`), and the corrected runs are in
  progress.

## 3. The deterministic guard: what is proven and what is measured

`galois_ml/guard.py` is the repository's first *constructive* component
rather than an instrument, so its claims need the same separation as the
rest of this note.

### Proven, in the strict sense

- **Attainability.** The Euler-factor formula fixes the set of values
  `a_(p^k)` can take for any Galois field of a given degree, and
  multiplicativity extends this to every index. A reported value outside
  that set is a proof that the channel lies. This is a theorem about the
  arithmetic, not a property of a model, a prompt, or a sample.
- **Soundness.** Honest tool output satisfies the theorem and the
  published grammar, so it is never flagged. Asserted by test across
  every task, and now also observed in live agent runs.
- **Determinism.** Guard verdicts are a function of the tool output, so
  per-channel detection is *computed* (`experiments/guard_matrix.py`),
  not estimated. There is no sampling error and no n to report.

### Proven only relative to an assumption

- **Schema checks** are complete against out-of-grammar corruption and
  silent against an attacker who conforms to the grammar. The
  `verification_stealth` channel exists to keep us from stating the
  stronger version: it inflates accuracy while emitting exactly one
  honest production, and it defeats every format check we have.
- **Recomputation** is provable given trusted local compute and data. If
  the orchestrator's own compute is compromised, the guarantee is void.
  This is the standard trusted-computing-base caveat and it is where our
  claim bottoms out.

### Not proven, and not claimed

Nothing here makes the *agents* trustworthy. The guard bounds what must
be trusted to a stochastic component and moves the remainder to one whose
behaviour is provable; agent conduct downstream of the guard remains a
measured quantity, in exact regret. In particular the guard does not
prevent an agent from reasoning badly about correctly-reported evidence,
and under our own restriction it cannot detect the `data` channel at all
(catching it would require class-distinguishing knowledge, which would
make the guard an oracle for the task and invalidate the measurement -
the same failure mode as the `theory_in_brief` redundancy that voided our
first poisoning runs).

### Relation to the external work above

SAGA-BFT and MUZZLE build architectures and defenses at the boundary we
declared out of scope (sandboxing, permissions, Byzantine agreement).
The guard operates at the *epistemic* boundary instead: it does not ask
whether a tool is authorized, it asks whether what the tool said can be
true. The two compose - an authorized tool can still report an
arithmetically impossible value, and no permission system detects that.

Our reading of the Anthropic Frontier Red Team finding is what motivated
the component. Their conclusion is that models abstractly understand
that sources have incentives but lack "a disposition to act on that
knowledge without prompting". Our inference, not their claim: a
disposition is the wrong place to put a check that is decidable. Where a
verification step is deterministic, assigning it to a stochastic
component is a design error rather than a training gap, and our 0%
detection rate on a channel refutable by one arithmetic comparison is
evidence for treating it that way.

## 4. Alignment with Anthropic, "Patterns and problems in multiagent
systems" (anthropic.com/research/multiagent-systems)

Read in full 2026-09-10. Their report is observational: it catalogues
failure patterns across coordination, conformity, epistemics, and
goal conflict. Ours is constructive on one slice of it. The
correspondences, with the honest gap on each:

| Their observation | Our result | Relationship |
|---|---|---|
| Closing thesis: models "abstractly understand that information sources have their own incentives ... What is missing is a disposition to act on that knowledge without prompting." | Flag policy: agents handed an explicit proof that evidence was false acknowledged it in 0/5 runs and did worse than unguarded. Repair and gate enforce without routing through disposition. | **Direct corroboration, and our design premise.** Their diagnosis is that the disposition is missing; our position is that a decidable check should not be assigned to a disposition at all. |
| Lie-detection: listener scored against a "trust everyone" naive policy and an oracle; "learn who lies" baseline excludes a source once contradicted by two others. | Our `credulous` role is the same naive policy; the ceiling is a theorem rather than a scripted oracle; our scouts experiments mirror the setup. | Same methodology; ours has a computable ceiling and unlimited instances. Their liar is stipulated; ours is derived. |
| "In principle detectable ... since a false report will eventually contradict an honest one" yet models underperform. | `impossible` channel: refutable from ONE tool output given facts in the brief; 0% agent detection across six generations; 100% deterministic detection. | Corroboration with a sharper instrument: detectability is exact, not eventual. |
| Conformity: 18/30 agents chose the identical branch name; agents "fail together." | 5 agents = 1.05 effective independent agents; errors 14--113x more correlated than independence. | Corroboration, quantified. Their implication ("isolated problems become systemic") is why our guarantees are N-independent. |
| Collusion persisted with all direct channels removed: agents "price-matched to the penny via a public listings board." | Our boundary result: coordination through legitimate, admissible task outputs cannot be prevented by content verification (the multi-agent analogue of label-permutation indistinguishability). | **Their empirical instance of our theoretical boundary.** Schema/type verification does not close it; we say so rather than claim otherwise. |
| Job-queue flooding: 2.4M requests, 117 accepted. | Out of scope: resource exhaustion; our runtime bounds wall clock only. | Honest non-overlap. Their finding is a reason our availability caveats matter. |
| Turf war: agents sabotage peers with self-replicating malware, disable accounts, camouflage processes. | `saboteur` condition (compromised pipeline member); release gate tolerates ALL N agents compromised for the released-risk bound. | Partial: we bound the released artifact's risk, not the agents' conduct toward each other or the host. Containment is SAGA/MUZZLE territory. |
| Hidden-profile: groups converge on shared evidence, discount the pivotal dissenter; "turning a simple dial ... will exacerbate the other." | Our layering is exactly a refusal to turn one dial: deterministic checks where decidable, gating where not, no global trust setting. | Alignment of design philosophy; we have no hidden-profile experiment. |
| Their prescription: mechanisms that "restructure the incentives around communication so that miscalibrated trust, in either direction, is caught and corrected." | Repair (caught and corrected, deterministically) and the gate (caught at release regardless of trust). | Our architecture is one such mechanism for the tool-evidence slice. |

What we should NOT claim from this alignment: that we address
coordination quality, goal conflict, or collusion; that our tasks
resemble theirs in realism; or that corroboration of the disposition
gap on a synthetic instrument replicates their findings. What we can
claim: their report identifies a gap ("disposition to act"), and our
paper is a constructive answer to one part of it that provably does
not depend on the gap being closed.

## Summary of the claim we are willing to defend

Agentic security research needs harm and defense-efficacy measured in
absolute terms, and that requires tasks whose correct answer is known in
closed form. This repository provides such tasks, plus instruments for
compromised agents, compromised tool channels, verification value, and
failure correlation. It is a controlled instrument for developing
measurement methodology, in the way small labelled corpora served early
adversarial-robustness work, and not a substitute for evaluation on
production agent deployments.

The constructive addition we are willing to defend on top of that: when a
verification step is decidable, it should be discharged by a bounded
deterministic component rather than delegated to an agent's judgment, and
this testbed can demonstrate the difference exactly - proven detection on
one side, a measured 0% on the other, with harm reduction in units of
regret against a provable ceiling. What we are not claiming is any
guarantee about the agents themselves.
