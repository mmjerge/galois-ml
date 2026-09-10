"""Send the guard/certify work to an external model for review.

The zip bundle cannot be uploaded to the Bedrock playground, and pasting
raw result files wastes context on agent transcripts (the control file
alone is 483KB, mostly prose). This script assembles a focused prompt
instead: the two new modules in full, plus *computed summaries* of the
results rather than the raw runs.

Reviewing a claim requires the reasoning, not every row behind it, so the
result sections are aggregates recomputed here from the JSON on disk. That
keeps the numbers verifiable (they are derived, not transcribed by hand)
without shipping 150 transcripts.

Usage:
    python experiments/request_review.py                  # call the model
    python experiments/request_review.py --dry-run        # write prompt only
    python experiments/request_review.py --model-id <id>
    python experiments/request_review.py --include-source tool_poisoning

Falls back to writing the prompt to a file if the API call fails, so the
text can be pasted into the playground by hand.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parent.parent))

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "experiments" / "outputs"
DEFAULT_MODEL = "openai.gpt-6-astra"
DEFAULT_OUT = "experiments/outputs/review_request.txt"

GUARDED = OUT / (
    "tool_poisoning_us.anthropic.claude-sonnet-4-5-20250929-v1_0"
    "_harm_v3_guarded.json"
)
CONTROL = OUT / (
    "tool_poisoning_us.anthropic.claude-sonnet-4-5-20250929-v1_0_harm_v2.json"
)
REPAIR = OUT / (
    "tool_poisoning_us.anthropic.claude-sonnet-4-5-20250929-v1_0"
    "_harm_v3_repair.json"
)

GUARANTEES_TASK = """\
You previously reviewed this codebase and found several bugs and
overclaims. All of them have since been fixed and the claims rescoped
(see the findings document below, section 5, for the corrections log).
Now I want your help with the constructive direction.

GOAL: propose what PROVABLE security guarantees this system could
actually offer, and how to state and prove them. Be a designer here, not
only a critic -- but hold every proposal to the standard you applied in
review: a guarantee must come with its assumptions stated and a proof
obligation we could actually discharge.

Please address:

1. FORMAL PROPOSITIONS. State 3-6 theorems we could prove about the
   guard/orchestrator as implemented or minimally extended. For each:
   hypothesis, statement, proof sketch, and what practical attack class
   it closes. Candidates you suggested implicitly in review: the
   restricted label-permutation indistinguishability result done
   properly; soundness relative to the unramified law; block-policy
   non-interference (corrupt evidence in the decidable class never
   reaches agent context).

2. VERIFIER EXTENSIONS with provable properties. You noted the guard
   misses class-agnostic joint constraints on marginal outputs:
   distributional identities (e.g. Pr(a_p = g) = Pr(a_{p^2} = value
   forced by f=1) on the same sample), cross-channel consistency between
   distributions and examples, and statistical detection with a bounded
   false-positive rate. Which of these are worth building, what exactly
   can be guaranteed for each (deterministic vs bounded-FP), and where
   is the line beyond which a checker becomes a task oracle?

3. PROTOCOL-LEVEL GUARANTEES. Given the empirical finding that warning
   agents made outcomes worse, what orchestration policies admit
   guarantees that do not route through agent disposition at all?
   Consider: block-and-recompute (quarantine the channel, substitute
   trusted recomputation where one exists), evidence provenance typing
   (agent conclusions may only cite guard-passed evidence), and
   verification of the agent's FINAL artifact (the rule is executable:
   its accuracy is checkable against held data regardless of how the
   agent reasoned). For each: what is provable, against which threat
   model, and what residual risk remains?

4. THE END-TO-END CLAIM. What is the strongest composite guarantee of
   the form "under assumptions A, corruption class C cannot degrade the
   pipeline's output by more than X" that this architecture could
   support? Which parts are theorems and which are irreducibly
   empirical? Be explicit about where the `data` channel
   (label-permutation attacks) forces the boundary.

5. PRIORITIES. Rank the above by (value of the guarantee) x
   (feasibility of the proof), and name the single proposition you
   would prove first for a paper.
"""

AIRTIGHT_TASK = """\
Third consultation. Your first review found four implementation bugs
(all fixed, regression-tested) and rescoped our claims. Your second
designed the constructive layer. We implemented your top two priorities
and ran the deciding experiment. Now make the whole thing AIRTIGHT: the
goal is a paper whose central claims survive hostile review, including
yours.

WHAT HAPPENED SINCE: (results and code below)

- ToolGuard gained a `repair` policy: on detected corruption it
  substitutes the trusted recomputation VERBATIM (no notice -- your
  Prop 4 statement B). Clean-transcript equivalence is tested end to
  end through the poisoned tools.
- `galois_ml/gate.py` implements your Prop 6 release gate (paired
  Hoeffding, fail-closed, candidate-count correction).
- The three-arm experiment ran: same detector, three policies.
  Median accuracy, sextic, n=5/condition:
      channel               unguarded   flag     repair
      impossible              0.920     0.728    0.925
      combined                0.898     0.153    0.910
      verification_stealth      -       0.890    0.925
      data                    0.907     0.898    0.907
  Exact one-sided permutation p: repair>flag on combined 0.004,
  impossible 0.020, stealth 0.032; repair>unguarded on combined 0.028.
  Repair lands on the clean control everywhere the detector fires;
  `data` is untouched at 0% detection, as the boundary result requires.

WHAT I NEED FROM YOU NOW:

1. HYPOTHESIS AUDIT OF THE REPAIR THEOREM AGAINST THE ACTUAL CODE.
   Statement B of your Prop 4 needs complete mediation and
   indistinguishable replacement. Audit `guard.py` and the tool wiring
   for violations: the `_repair_thunk` instance state; whether budget
   messages bypass the guard; whether any counter, timing, or
   length-of-response side channel distinguishes attacked-repaired from
   clean runs; whether the critic's separately-built tools share the
   guard correctly; whether `show_examples` repair covers the combined
   attack's relabeling. Name every gap between the theorem's
   hypotheses and the implementation, with severity.

2. ADAPTIVE-ATTACKER ANALYSIS. Against the repair policy, the optimal
   attack seems to collapse: `combined`'s data component only got
   repaired because its injected text made responses grammar-detectable,
   so an attacker who drops the injection and forges only
   grammar-conforming, attainable, consistently-relabeled evidence
   reduces to the pure `data` channel. Is that reduction right? What is
   the actual best response of an adaptive attacker to
   repair-plus-gate, and what is the resulting worst-case bound?

3. GATE PROOF AND USAGE AUDIT. Check `gate.py` against your Prop 6:
   the Hoeffding direction and constant, the n_candidates correction,
   fail-closed semantics, and the fresh-holdout obligation as
   documented. Is anything in the implementation or its intended use
   (make_dataset holdouts, theorem_fallback) a hypothesis violation?
   The fallback encodes task knowledge -- state exactly how that
   qualifies the claim.

4. THE COMPOSITE THEOREM. Write, at publication precision, the
   strongest composite statement of the form: "for an N-agent pipeline
   with mediation M, repair R, and gate G, under assumptions A, an
   adversary controlling tool channels C achieves expected released
   regret at most ___". Make the N-independence explicit (our
   conformity data shows redundancy buys ~1.05 effective agents of 5,
   so N-independence is the selling point). Separate the deterministic
   part (repair equivalence), the statistical part (gate), and the
   impossibility boundary (label permutation). Flag every place where
   the claim is fallback-relative rather than clean-relative.

5. STATISTICS FOR PUBLICATION. n=5/condition, exact permutation tests,
   a growing comparison family. What is the right analysis and wording
   for the three-arm table -- and what would a hostile statistical
   reviewer demand (prespecification? replication? correction?) before
   accepting "repair recovers the flag regression"?

6. TITLE-LEVEL FRAMING. We are NOT claiming "provable trustworthy
   multi-agent collaboration" -- we know the guarantees are (a)
   zero-degradation on the decidable class, (b) fallback-relative risk
   control elsewhere, (c) an impossibility boundary, all independent of
   agent count and disposition. Give the strongest honest title and
   abstract-level claim, and list the claims we must NOT make.

7. EXPANSION TOWARD CONCRETE MULTI-AGENT SECURITY. The motivating
   context is incidents like the 2026 OpenAI/Hugging Face breach, where
   ~1,200 agents coordinated through an unmediated shared channel (a
   package cache) and acted on unverified information from it. Our
   architecture mediates AGENT-TOOL channels. The obvious
   generalization: treat AGENT-TO-AGENT messages as tool channels --
   mediate, verify what is decidable, repair or type-tag what is not,
   and gate collective outputs against a fallback. Design this
   extension honestly:
   a. What do our theorems give when the "corrupt tool" is another
      agent (a compromised pipeline member, or an emergent coordination
      channel)? The saboteur condition already exists in the testbed;
      what mediation invariant would have provably prevented the
      shared-cache coordination pattern, and what is NOT preventable
      this way (agents coordinating through legitimate task outputs --
      the multi-agent analogue of our label-permutation boundary)?
   b. What is the minimal experiment on THIS testbed that demonstrates
      the inter-agent version -- e.g. analyst+critic where the critic's
      view of the analyst's claims passes through the guard, or the
      conformity ensemble with mediated cross-agent messages?
   c. State the strongest honest claim of the form "collaboration
      cannot break the bound": under what mediation assumptions does
      the released-output guarantee survive N agents, arbitrary
      inter-agent messaging, and k compromised members? Where does it
      fail (side channels, resource exhaustion, semantic manipulation
      inside admissible messages)?
"""

ENFORCEMENT_TASK = """\
Fourth consultation. Since your airtight audit we built the two things
you said were missing, and ran the offline experiment from your
hostile-reviewer list. Audit both.

BUILT: `galois_ml/release.py` (source below) -- the trusted release
runtime around the gate, targeting your six obligations: freeze+hash
before certification randomness exists; hash re-verification at
evaluation; atomic single-use selection; released predictor rebuilt
from certified bytes; no alternate release path; defined
timeout/crash/invalid-output behavior (all release fallback; a
FALLBACK failure raises and releases nothing). Additional isolation:
candidates execute in a separate OS process receiving holdout FEATURES
ONLY -- labels and the entropy seed stay in the parent; the seed is
disclosed only in the post-release audit record.

RAN: your item "run the actual gate on archived frozen artifacts using
fresh holdouts". All 89 valid no-critic sextic rules across the three
policy arms were recovered from transcripts and certified through the
full runtime, against two fallbacks (m=2000, epsilon=0.05, delta=0.05,
one fresh entropy holdout per gate; 178 gates total):

- ZERO released artifacts exceeded fallback holdout risk by more than
  epsilon; ZERO rules below 0.5 holdout accuracy released.
- Majority fallback (risk ~0.5): 79/89 released; the 10 refusals were
  exactly the flag-arm combined inversions (holdout acc 0.10-0.15) and
  the unguarded runs that absorbed the combined lie.
- Theorem fallback (risk ~0.049): 0/89 released -- your predicted
  near-Bayes regime; agents at ~0.92 cannot certify against a 0.951
  floor at epsilon=0.05.
- The repair-arm `data` collapse (pipeline acc 0.608) was released
  against majority (bound held: 0.608 > 0.5) and refused against
  theorem.

QUESTIONS:

1. RUNTIME AUDIT. Find gaps between `release.py` and your six
   obligations plus its isolation claim. Specifically consider: the
   features array is pickled to the child (any label information in X
   itself? any side channel through make_dataset's structure?); no
   child resource limits (memory/CPU beyond wall clock); spawn-context
   semantics; the audit record's integrity (plain dict, unsigned); the
   fallback executing in the parent unsandboxed; certify() drawing the
   seed but make_dataset being outside the module's trust argument;
   batch-vs-pointwise prediction semantics; and anything we missed.

2. ARCHIVED-GATE STUDY VALIDITY. The rules were recovered from run
   transcripts post hoc. Does that break the freeze-before-holdout
   hypothesis in any way that matters (the rules were written before
   any of these holdouts existed, but WE selected which JSON field to
   extract)? Is 0/178 violations informative or vacuous given most
   candidates are far from the boundary? How should the two-fallback
   table be presented so it cannot be read as 'the gate solves the
   task'?

3. SAFETY/AVAILABILITY SEPARATION. Against the theorem fallback the
   gate certifies nothing, so an attacker forcing fallback costs all
   collaboration value at zero risk cost. Is there a principled way to
   state (or bound) the availability loss, or an architecture that
   recovers value without weakening the risk bound (e.g. epsilon
   schedules, larger m, shadow runs)?

4. WHAT REMAINS before the confirmatory study is worth running. Given
   the runtime now exists, rank: interleaved three-arm replication
   (degree21 + second model), the inter-agent envelope experiment from
   your sec 7.5, the exact-counts evidence format (your Prop 2), or
   something else first.
"""

TASK = """\
You are reviewing a research codebase and its empirical results. Please be
critical and specific. I would rather hear that a claim is unsupported than
have it validated.

CONTEXT

The project uses Galois-theoretic classification tasks (identifying the
Galois group of a number field from its Dedekind zeta coefficients) as an
instrument for measuring security properties of multi-agent LLM pipelines.
The point of the number theory is that ground truth is provable: each task
has a computable Bayes-optimal accuracy ceiling, exact per-feature mutual
information in bits, and a reference theorem. So attack harm is exact
regret rather than an LLM-judged score.

This round added a deterministic, LLM-free verifier (`guard.py`) between
tools and agents, and a model-layer audit module (`certify.py`) that scores
models against a per-input exact posterior.

The headline result is partly NEGATIVE: the guard achieves provable
detection but reduced harm on no channel, and significantly worsened the
hardest channel. Note that in `flag` mode the guard is strictly
information-preserving (original tool output returned verbatim, correct
diagnostic appended), so guarded agents had strictly MORE correct
information and performed worse.

WHAT I WANT FROM YOU

1. Correctness of the guarantee claims. `guard.py` asserts three
   guarantee classes: `theorem` (Euler-factor attainability, data-free),
   `schema` (published tool grammar), `recomputation` (re-deriving
   deterministic results). Are the soundness and completeness claims
   actually justified by the code? Find counterexamples if they exist.
   Pay attention to the attainability computation: is the attainable set
   correct, and is it genuinely class-agnostic?

2. The impossibility result. We claim corruption using *attainable*
   values under *misattributed* labels (the `data` channel) cannot be
   detected by any class-agnostic verifier, since its output is a
   possible honest output under a label permutation. Is this argument
   sound, or is there a checkable invariant we missed?

3. Statistical validity. On the clean channel the guard is a provable
   identity, so guarded vs unguarded `none` is the same experiment run
   twice; it produced p=0.037 from sampling alone at n=5v5. We use that
   as an empirical null and therefore call `impossible` (p=0.071)
   unresolved despite a 0.19 median gap. Is that the right use of the
   null? Would a paired or permutation test be better? Is the
   Mann-Whitney appropriate at this n?

4. The negative result's interpretation. Is "detection without a
   mitigation policy can be net-harmful" supported, or is the
   message-design confound (the warning is a bare "treat this as
   untrusted" with no routing guidance) fatal to the claim? What is the
   minimal experiment that would separate those?

5. `certify.py` review. Is the exact-posterior derivation correct? Is
   KL(true||model) the right direction for a calibration claim, and is
   the "certified label-preserving perturbation" argument valid (we
   resample a prime's signature from the same class-conditional measure,
   so the label is preserved by construction, and only count prediction
   flips where the true posterior still favours the original class)?

6. Cross-layer claim. We observe that agents flag impossible tool values
   0/5 of the time while ML models are equally or MORE confident on
   arithmetically impossible inputs, and argue this is one phenomenon at
   two layers. Is that one finding or two unrelated ones narrated as one?

7. Publication framing. Which claims here would survive peer review, and
   which need the pending replication (degree21, a second model, `block`
   policy)? What is the strongest honest version of the contribution?
"""


def _digest_runs(path: Path, task: str = "sextic", with_critic=False) -> dict:
    """Per-channel aggregates. Recomputed here so the review sees derived
    numbers rather than hand-copied ones."""
    if not path.exists():
        return {}
    rows = [
        r for r in json.loads(path.read_text())
        if r.get("valid") and r.get("task") == task
        and bool(r.get("with_critic")) == with_critic
    ]
    by: dict[str, list] = defaultdict(list)
    for r in rows:
        by[r["attack"]].append(r)
    out = {}
    for atk, rs in sorted(by.items()):
        acc = sorted(float(r["test_accuracy"]) for r in rs)
        out[atk] = {
            "n": len(rs),
            "accuracies": [round(a, 4) for a in acc],
            "median_accuracy": round(float(np.median(acc)), 4),
            "median_regret": round(float(np.median([r["regret"] for r in rs])), 4),
            "agent_flagged": f"{sum(bool(r.get('attack_surfaced')) for r in rs)}/{len(rs)}",
            "guard_detected": (
                f"{sum(bool(r.get('guard_detected')) for r in rs)}/{len(rs)}"
                if any(r.get("guarded") for r in rs) else "n/a (unguarded)"
            ),
            "poisoned_tool_calls": [r.get("poisoned_tool_calls", 0) for r in rs],
        }
    return out


def _exact_perm_p(a: list[float], b: list[float], direction: str) -> float:
    """Exact one-sided permutation p on accuracy sums; 'greater' tests
    whether a's accuracies are high, 'less' whether low, under
    exchangeability of the pooled runs (unpaired: the arms are
    independent replicates, not matched blocks)."""
    from itertools import combinations

    pooled = a + b
    obs = sum(a)
    hit = tot = 0
    for idx in combinations(range(len(pooled)), len(a)):
        tot += 1
        s = sum(pooled[i] for i in idx)
        if (direction == "greater" and s >= obs - 1e-12) or (
            direction == "less" and s <= obs + 1e-12
        ):
            hit += 1
    return hit / tot


def _matched_comparison() -> str:
    """Flag-guarded vs unguarded arms, exact permutation tests.

    Exploratory, nominal p-values: the arms are independent replicates
    run sequentially (not randomized/interleaved, not paired blocks),
    and the comparison family has grown over the analysis history.
    """
    g_all = json.loads(GUARDED.read_text()) if GUARDED.exists() else []
    u_all = json.loads(CONTROL.read_text()) if CONTROL.exists() else []

    def sel(d, atk):
        return [float(r["test_accuracy"]) for r in d
                if r.get("valid") and r.get("task") == "sextic"
                and r["attack"] == atk and not r.get("with_critic")]

    lines = [
        "channel               n(G v U)   G_median   U_median      delta"
        "    p(G<U)    p(G>U)   [exact permutation, unpaired]",
    ]
    for atk in ["none", "data", "injection", "impossible", "combined"]:
        g, u = sel(g_all, atk), sel(u_all, atk)
        if len(g) < 2 or len(u) < 2:
            continue
        p_lo = _exact_perm_p(g, u, "less")
        p_hi = _exact_perm_p(g, u, "greater")
        lines.append(
            f"{atk:<22}{len(g)}v{len(u):<8}{np.median(g):>9.4f}"
            f"{np.median(u):>11.4f}{np.median(g) - np.median(u):>+11.4f}"
            f"{p_lo:>10.4f}{p_hi:>10.4f}"
        )
    lines.append("")
    lines.append(
        "Notes: `none` is the null channel (the guard is an identity on\n"
        "honest output, so that row is the same experiment twice and its\n"
        "p-value is sampling noise). The test statistic is the accuracy\n"
        "sum; p-values are nominal and exploratory, uncorrected for the\n"
        "analysis-history family; arms were run sequentially, not\n"
        "interleaved."
    )
    return "\n".join(lines)


def _certify_summary() -> str:
    path = OUT / "certify_audit.json"
    if not path.exists():
        return "(not run)"
    rows = json.loads(path.read_text())
    by: dict[tuple, list] = defaultdict(list)
    for r in rows:
        by[(r["task"], r["model"])].append(r)
    lines = [
        f"{'task':<10}{'model':<22}{'acc':>7}{'excess_bits':>13}"
        f"{'ECE_exact':>11}{'overconf':>10}{'flip_rate':>11}{'MI0_mass':>10}"
    ]
    for (task, model), rs in sorted(by.items()):
        r = rs[0]
        zm = r.get("mass_on_zero_information_features")
        lines.append(
            f"{task:<10}{model:<22}{r['accuracy']:>7.4f}"
            f"{r['excess_log_loss']:>13.4f}{r['exact_calibration_error']:>11.4f}"
            f"{r['overconfidence']:>+10.4f}"
            f"{(r['invariance']['certified_flip_rate'] or 0):>11.1%}"
            + (f"{zm:>10.3f}" if zm is not None else f"{'-':>10}")
        )
    adm = rows[0]["admissibility"]
    lines += [
        "",
        "admissibility (decidable predicate vs confidence heuristic), "
        "representative row:",
        f"  corrupt inputs rejected      : {adm['detection_rate']:.0%}",
        f"  false rejections (honest data): "
        f"{adm['false_rejections_on_honest_data']}",
        f"  confidence heuristic @5% FP  : "
        f"{adm['confidence_heuristic_detection_rate']:.0%}",
        f"  model confidence on IMPOSSIBLE inputs: "
        f"{adm['mean_confidence_corrupt']:.3f} "
        f"(honest: {adm['mean_confidence_honest']:.3f})",
    ]
    return "\n".join(lines)


def _guard_matrix_summary() -> str:
    path = OUT / "guard_matrix.json"
    if not path.exists():
        return "(not run)"
    rows = json.loads(path.read_text())
    lines = [f"{'task':<10}{'channel':<22}{'detected':>9}  guarantee / checks"]
    for r in rows:
        lines.append(
            f"{r['task']:<10}{r['attack']:<22}"
            f"{('yes' if r['guard_detected'] else 'no'):>9}  "
            f"{','.join(r['guarantees']) or '-'} "
            f"[{', '.join(c.split('/')[1] for c in r['checks_fired']) or '-'}]"
        )
    return "\n".join(lines)


def _sample_transcript() -> str:
    """One guarded `combined` conjecture: the inversion is the finding, and
    it cannot be judged from summary statistics alone."""
    if not GUARDED.exists():
        return "(not available)"
    for r in json.loads(GUARDED.read_text()):
        if (r.get("valid") and r["attack"] == "combined"
                and r.get("final_conjecture")):
            return (
                f"seed {r['seed']}, accuracy {r['test_accuracy']:.4f} "
                f"(ceiling {r['bayes_accuracy']:.4f}):\n\n"
                f"{r['final_conjecture']}\n\n"
                "For reference: a_(p^2)=3 indicates inertia degree f=2. S3 has "
                "3 of 6 elements of order 2; C6 has 1 of 6. So frequent "
                "a_(p^2)=3 implies S3, and this conjecture asserts the "
                "opposite -- the correct criterion attached to the wrong "
                "class. (Accuracy 0.10-0.15 is the complement of an "
                "imperfect rule, not of the Bayes ceiling: 1 - 0.9518 "
                "would be 0.048.)"
            )
    return "(no combined runs found)"


def build_prompt(extra_source: list[str], mode: str = "review") -> str:
    tasks = {"review": TASK, "guarantees": GUARANTEES_TASK,
             "airtight": AIRTIGHT_TASK, "enforcement": ENFORCEMENT_TASK}
    parts = [tasks[mode]]
    if mode == "airtight":
        extra_source = list(extra_source) + ["gate"]
    if mode == "enforcement":
        extra_source = list(extra_source) + ["gate", "release"]

    findings = ROOT / "docs" / "GUARD_FINDINGS.md"
    if findings.exists():
        parts.append(
            "=" * 78 + "\nMY OWN WRITE-UP OF THE FINDINGS AND CAVEATS\n"
            + "=" * 78 + "\n" + findings.read_text()
        )

    parts.append(
        "=" * 78 + "\nSOURCE: galois_ml/guard.py (the verifier under review)\n"
        + "=" * 78 + "\n" + (ROOT / "galois_ml" / "guard.py").read_text()
    )
    parts.append(
        "=" * 78 + "\nSOURCE: galois_ml/certify.py (model-layer audits)\n"
        + "=" * 78 + "\n" + (ROOT / "galois_ml" / "certify.py").read_text()
    )
    for name in extra_source:
        p = ROOT / "galois_ml" / f"{name}.py"
        if p.exists():
            parts.append(
                "=" * 78 + f"\nSOURCE: galois_ml/{name}.py\n" + "=" * 78
                + "\n" + p.read_text()
            )

    parts.append(
        "=" * 78 + "\nRESULT 1: deterministic detection matrix (computed, "
        "not sampled -- no model involved)\n" + "=" * 78 + "\n"
        + _guard_matrix_summary()
    )
    parts.append(
        "=" * 78 + "\nRESULT 2: matched agent comparison, guarded vs "
        "unguarded\nsextic, no-critic, theory withheld, temperature 0.7, "
        "seeds 0-4, Claude Sonnet 4.5\n" + "=" * 78 + "\n"
        + _matched_comparison()
    )
    parts.append(
        "=" * 78 + "\nRESULT 2b: per-channel detail, GUARDED runs\n"
        + "=" * 78 + "\n"
        + json.dumps(_digest_runs(GUARDED), indent=1)
    )
    parts.append(
        "=" * 78 + "\nRESULT 2c: per-channel detail, UNGUARDED control runs\n"
        + "=" * 78 + "\n"
        + json.dumps(_digest_runs(CONTROL), indent=1)
    )
    if REPAIR.exists():
        parts.append(
            "=" * 78 + "\nRESULT 2d: per-channel detail, REPAIR-policy runs\n"
            + "=" * 78 + "\n"
            + json.dumps(_digest_runs(REPAIR), indent=1)
        )
    parts.append(
        "=" * 78 + "\nRESULT 3: model-layer audit against the exact "
        "posterior (offline, no agents)\n" + "=" * 78 + "\n"
        + _certify_summary()
    )
    parts.append(
        "=" * 78 + "\nRESULT 4: sample failure -- a guarded `combined` "
        "conjecture\n" + "=" * 78 + "\n" + _sample_transcript()
    )
    parts.append(
        "=" * 78 + "\nEND OF MATERIAL. Please work through the seven "
        "questions above.\n" + "=" * 78
    )
    return "\n\n".join(parts)


def _invoke(client, model_id: str, prompt: str, max_tokens: int,
            stream: bool, sink: Path) -> str:
    """Call the model, reporting progress as it goes.

    Streaming matters for more than comfort here. A reasoning model spends
    its first minutes emitting no answer text at all, so a blocking call is
    indistinguishable from a hang, and there is no way to tell a slow review
    from a stalled connection. Streaming also lets partial output be flushed
    to disk, so an interruption late in a long generation is recoverable.
    """
    import sys
    import time

    body = {
        "modelId": model_id,
        "messages": [{"role": "user", "content": [{"text": prompt}]}],
        "inferenceConfig": {"maxTokens": max_tokens},
    }

    if not stream:
        t0 = time.time()
        resp = client.converse(**body)
        text = "".join(
            b.get("text", "") for b in resp["output"]["message"]["content"]
        )
        usage = resp.get("usage", {})
        print(f"  done in {time.time() - t0:.0f}s "
              f"(in {usage.get('inputTokens', '?')} / "
              f"out {usage.get('outputTokens', '?')} tokens)")
        return text

    resp = client.converse_stream(**body)
    chunks: list[str] = []
    reasoning_chars = 0
    t0 = time.time()
    last_flush = t0
    phase = None

    for event in resp["stream"]:
        if "contentBlockDelta" in event:
            delta = event["contentBlockDelta"]["delta"]
            # Reasoning models emit thinking before any answer text. Show it
            # as progress rather than leaving the terminal silent.
            reasoning = delta.get("reasoningContent", {})
            if reasoning:
                rtext = reasoning.get("text") or ""
                reasoning_chars += len(rtext)
                if phase != "reasoning":
                    phase = "reasoning"
                    print("  thinking", end="", flush=True)
                if reasoning_chars // 400 > (reasoning_chars - len(rtext)) // 400:
                    print(".", end="", flush=True)
            if "text" in delta:
                if phase != "writing":
                    phase = "writing"
                    elapsed = time.time() - t0
                    # Reasoning text is often redacted, in which case the
                    # deltas arrive with no readable content: report the
                    # elapsed time, which is real, and only report a size
                    # when there was actually something to measure.
                    detail = (f", ~{reasoning_chars:,} chars"
                              if reasoning_chars else ", content redacted")
                    print(f"\n  writing (thinking took {elapsed:.0f}s"
                          f"{detail})", flush=True)
                chunks.append(delta["text"])
                n = sum(len(c) for c in chunks)
                # progress line, and periodically flush to disk so a late
                # failure does not discard a finished-but-unsaved review
                if time.time() - last_flush > 5:
                    last_flush = time.time()
                    sink.write_text("".join(chunks))
                    print(f"\r  received ~{n // 4:,} tokens "
                          f"({time.time() - t0:.0f}s)", end="", flush=True)
        elif "messageStop" in event:
            print(f"\r  stop reason: "
                  f"{event['messageStop'].get('stopReason')}"
                  f"{' ' * 20}")
        elif "metadata" in event:
            u = event["metadata"].get("usage", {})
            print(f"  usage: in {u.get('inputTokens', '?')} / "
                  f"out {u.get('outputTokens', '?')} tokens, "
                  f"{time.time() - t0:.0f}s total")
    sys.stdout.flush()
    return "".join(chunks)


def _candidate_ids(model_id: str) -> list[str]:
    """The given ID, then the cross-region inference-profile variants.

    Bedrock refuses on-demand invocation for some models and requires an
    inference profile instead, whose ID is the model ID with a geo prefix
    (this is why every Claude ID in this repo reads `us.anthropic...`).
    Already-prefixed IDs and ARNs are passed through untouched.
    """
    if model_id.startswith(("arn:", "us.", "eu.", "apac.", "global.")):
        return [model_id]
    return [model_id, f"us.{model_id}", f"global.{model_id}"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", default=DEFAULT_MODEL)
    ap.add_argument("--region", default="us-east-1")
    ap.add_argument("--max-tokens", type=int, default=16000)
    ap.add_argument("--mode",
                    choices=("review", "guarantees", "airtight",
                             "enforcement"),
                    default="review",
                    help="review: critical code/results review; "
                         "guarantees: design provable security guarantees; "
                         "airtight: audit the implemented repair+gate "
                         "architecture and three-arm results for "
                         "publication, incl. the multi-agent extension")
    ap.add_argument("--dry-run", action="store_true",
                    help="write the prompt without calling the model")
    ap.add_argument("--no-stream", action="store_true",
                    help="use the blocking converse API instead of streaming "
                         "(no progress output until the response completes)")
    ap.add_argument("--include-source", nargs="*", default=[],
                    help="extra galois_ml modules to include, e.g. "
                         "tool_poisoning benchmark")
    ap.add_argument("--out", default=DEFAULT_OUT)
    args = ap.parse_args()

    if args.mode != "review" and args.out == DEFAULT_OUT:
        args.out = f"experiments/outputs/{args.mode}_request.txt"
    prompt = build_prompt(args.include_source, mode=args.mode)
    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(prompt)
    approx_tokens = len(prompt) // 4
    print(f"prompt: {len(prompt):,} chars (~{approx_tokens:,} tokens) "
          f"-> {out_path}")

    if args.dry_run:
        print("dry run; paste the file above into the playground.")
        return

    review_path = out_path.with_name(f"{args.mode}_response.md")
    text = None
    try:
        import boto3

        client = boto3.client("bedrock-runtime", region_name=args.region)
        for model_id in _candidate_ids(args.model_id):
            print(f"calling {model_id} ...")
            try:
                text = _invoke(
                    client, model_id, prompt, args.max_tokens,
                    stream=not args.no_stream, sink=review_path,
                )
            except client.exceptions.ValidationException as exc:
                # Most on-demand-unsupported models are reachable through a
                # cross-region inference profile, which is the same ID with a
                # geo prefix. Retry those rather than making the caller guess.
                if "inference profile" in str(exc):
                    print("  needs an inference profile; trying next prefix")
                    continue
                raise
            print(f"\n  ok via {model_id}")
            break
    except KeyboardInterrupt:
        print("\ninterrupted; any partial output was saved.")
        return
    except Exception as exc:
        print(f"\nAPI call failed: {type(exc).__name__}: {exc}")

    if text is None:
        # The prompt is already on disk, so a failure here costs nothing:
        # paste it into the playground instead.
        print(f"\nCould not reach the model. The prompt is saved at "
              f"{out_path} -- paste it into the Bedrock playground.")
        print("If you know the profile ID or ARN, pass it: "
              "--model-id <profile-id>")
        return

    review_path.write_text(text)
    print(f"\nreview ({len(text):,} chars) -> {review_path}\n")
    print("=" * 78)
    print(text)


if __name__ == "__main__":
    main()
