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

## Summary of the claim we are willing to defend

Agentic security research needs harm and defense-efficacy measured in
absolute terms, and that requires tasks whose correct answer is known in
closed form. This repository provides such tasks, plus instruments for
compromised agents, compromised tool channels, verification value, and
failure correlation. It is a controlled instrument for developing
measurement methodology, in the way small labelled corpora served early
adversarial-robustness work, and not a substitute for evaluation on
production agent deployments.
