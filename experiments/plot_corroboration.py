"""Side-by-side: Anthropic's reported aggregation failure vs ours.

Produces experiments/figures/fig_corroboration.png with three panels:

  A. Anthropic Frontier Red Team, reported hidden-profile result: groups
     of four agents decide between two options where the shared evidence
     favours the wrong one and private information is decisive. Group
     accuracy lands far below the solo ceiling.
  B. This work: majority vote over five independent agents, against the
     accuracy an ensemble with the same individual error rates would
     reach if failures were independent, and against the provable Bayes
     ceiling.
  C. This work: why B fails -- error-overlap correlation relative to
     independence, before and after removing irreducible task ambiguity.

Attribution and honesty constraints observed here:

* Panel A does NOT reproduce Anthropic's figure. The values are read from
  the prose of their report ("Mythos 5 groups scored about 85%; other
  models scored 17-36%, far below solo ceilings near 100%") and are
  therefore approximate, range-based, and labelled as such.
* The two results share a theme but not a mechanism, and the caption says
  so: their agents hold *different* private facts and fail to surface or
  trust them; our agents see the *same* data and fail on the same items.
  Both are aggregation underdelivering against its own ceiling.

Source: "Patterns and problems in emerging multiagent systems",
Anthropic Frontier Red Team, 2026-08-13.
https://www.anthropic.com/research/multiagent-systems

Run:  python experiments/plot_corroboration.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

# Make the repo importable regardless of interpreter or install state.
import pathlib as _pathlib
sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parent.parent))

OUT = Path("experiments/outputs")
FIGS = Path("experiments/figures")

#: Values quoted from the Anthropic report's prose, not digitized from
#: their figure. Ranges are given where the report gives a range.
ANTHROPIC_HIDDEN_PROFILE = {
    "source": "Anthropic FRT 2026-08-13 (values from report text)",
    "solo_ceiling": 1.00,          # "solo ceilings near 100%"
    "best_model_group": 0.85,      # "Mythos 5 groups scored about 85%"
    "other_models_range": (0.17, 0.36),  # "other models scored 17-36%"
    "n_episodes": 400,             # "n=400 episodes per model"
}


def load_our_conformity() -> list[dict]:
    """Our conformity metrics, newest file, tasks with measurable error."""
    rows = []
    for path in sorted(OUT.glob("conformity_*.json")):
        for res in json.loads(path.read_text()):
            m = res.get("metrics", {})
            if not m or "note" in m:
                continue
            # skip degenerate tasks where every agent was perfect: there
            # is no aggregation gap to show
            if m.get("correlation_ratio") is None:
                continue
            rows.append({
                "task": res["task"],
                "ceiling": res["bayes_accuracy"],
                "individual_mean": m["individual_accuracy_mean"],
                "majority": m["majority_vote_accuracy"],
                "independent": m["independent_ensemble_accuracy"],
                "n_agents": m["n_valid_agents"],
                "corr_raw": m["correlation_ratio"],
                "corr_avoidable": m.get("avoidable_correlation_ratio"),
                "effective": m["effective_independent_agents"],
                "distinct": m["distinct_conjectures"],
            })
    return rows


def main() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ours = load_our_conformity()
    if not ours:
        print("no non-degenerate conformity results yet; run "
              "experiments/conformity_eval.py first")
        return

    FIGS.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.8))
    a, b, c = axes

    # ---- Panel A: Anthropic reported hidden-profile ------------------
    ap = ANTHROPIC_HIDDEN_PROFILE
    lo, hi = ap["other_models_range"]
    labels = ["most models\n(group)", "best model\n(group)", "solo\nceiling"]
    values = [(lo + hi) / 2, ap["best_model_group"], ap["solo_ceiling"]]
    errs = [[(lo + hi) / 2 - lo, 0, 0], [hi - (lo + hi) / 2, 0, 0]]
    colors = ["#C44E52", "#DD8452", "#4C72B0"]
    a.bar(labels, values, color=colors, yerr=errs, capsize=4)
    a.axhline(ap["solo_ceiling"], ls="--", lw=1, color="#4C72B0")
    a.set_ylim(0, 1.12)
    a.set_ylabel("accuracy")
    a.set_title("A. Anthropic FRT (reported)\nhidden-profile: groups of 4 agents",
                fontsize=10)
    a.text(0.5, -0.30,
           "values read from report text, not digitized;\n"
           "groups fall far short of the solo ceiling",
           transform=a.transAxes, ha="center", fontsize=8, color="#444")

    # ---- Panel B: our aggregation gap --------------------------------
    tasks = [r["task"] for r in ours]
    x = np.arange(len(tasks))
    width = 0.26
    b.bar(x - width, [r["individual_mean"] for r in ours], width,
          label="mean single agent", color="#8172B3")
    b.bar(x, [r["majority"] for r in ours], width,
          label="majority vote (observed)", color="#C44E52")
    b.bar(x + width, [r["independent"] for r in ours], width,
          label="vote if failures independent", color="#55A868")
    for i, r in enumerate(ours):
        b.hlines(r["ceiling"], i - 0.42, i + 0.42, color="black", lw=1.6,
                 linestyles="--")
        b.annotate("", xy=(i, r["independent"]), xytext=(i, r["majority"]),
                   arrowprops=dict(arrowstyle="<->", color="#333", lw=1.2))
        b.text(i + 0.03, (r["majority"] + r["independent"]) / 2,
               "unrealized\ngain", fontsize=7.5, va="center", color="#333")
    b.set_xticks(x, tasks)
    b.set_ylim(0.80, 1.02)
    b.set_ylabel("accuracy")
    b.set_title("B. This work (n=5 agents/task)\nvoting does not deliver its "
                "theoretical gain", fontsize=10)
    b.legend(fontsize=7.5, loc="lower right")
    b.text(0.5, -0.30,
           "dashed line = provable Bayes ceiling;\n"
           "effective independent agents: "
           + ", ".join(f"{r['task']} {r['effective']:.2f}/{r['n_agents']}"
                       for r in ours),
           transform=b.transAxes, ha="center", fontsize=8, color="#444")

    # ---- Panel C: the mechanism -------------------------------------
    raw = [r["corr_raw"] for r in ours]
    avoid = [r["corr_avoidable"] if r["corr_avoidable"] else np.nan
             for r in ours]
    c.bar(x - 0.18, raw, 0.36, label="all errors", color="#937860")
    c.bar(x + 0.18, avoid, 0.36, label="avoidable errors only",
          color="#DA8BC3")
    c.axhline(1.0, color="black", lw=1.2, ls="--")
    c.text(len(tasks) - 0.5, 1.6, "independence", fontsize=8, ha="right")
    for i, (rv, av) in enumerate(zip(raw, avoid)):
        c.text(i - 0.18, rv + 2, f"{rv:.0f}x", ha="center", fontsize=8)
        if not np.isnan(av):
            c.text(i + 0.18, av + 2, f"{av:.0f}x", ha="center", fontsize=8)
    c.set_xticks(x, tasks)
    c.set_ylabel("error-overlap / overlap if independent")
    c.set_title("C. This work: why B fails\nagents fail on the SAME items",
                fontsize=10)
    c.legend(fontsize=7.5)
    c.text(0.5, -0.30,
           "removing irreducible task ambiguity raises the ratio:\n"
           "shared blind spots, not task difficulty",
           transform=c.transAxes, ha="center", fontsize=8, color="#444")

    fig.suptitle(
        "Multi-agent aggregation underdelivers against its own ceiling: "
        "an independent corroboration with provable ground truth",
        fontsize=11.5, y=1.02,
    )
    fig.tight_layout()
    path = FIGS / "fig_corroboration.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"wrote {path}")
    print("\nNote on comparability: the two results share a theme but not a")
    print("mechanism. Anthropic's agents hold different private facts and")
    print("fail to surface or trust them; ours see identical data and fail")
    print("on the same items. Both show aggregation falling short of what")
    print("the information available to the group would permit.")
    for r in ours:
        print(f"\n  {r['task']}: {r['distinct']}/{r['n_agents']} distinct "
              f"conjectures yet {r['corr_raw']:.0f}x error correlation")
        print(f"    majority {r['majority']:.4f} vs independent "
              f"{r['independent']:.4f} (ceiling {r['ceiling']:.4f})")


if __name__ == "__main__":
    main()
