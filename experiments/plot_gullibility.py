"""Our gullibility curve beside Anthropic's reported one.

Writes experiments/figures/fig_gullibility.png with two panels drawn in
the same idiom:

  Left   Anthropic Frontier Red Team's reported curve: routing-decision
         accuracy against how often an untrustworthy scout lies,
         bracketed by "trust everyone" and "learns who lies".
  Right  This work: accuracy against the fraction of tool calls returning
         falsified data, bracketed by a measured credulous agent and the
         provable Bayes ceiling.

Attribution: the left panel does NOT reproduce their image. Values are
read off the published figure and are approximate; they are plotted as
open markers with a dashed style and labelled as digitized so they are
never mistaken for our measurements.

Source: "Patterns and problems in emerging multiagent systems", Anthropic
Frontier Red Team, 2026-08-13.
https://www.anthropic.com/research/multiagent-systems

Run:  python experiments/plot_gullibility.py
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

# Make the repo importable regardless of interpreter or install state.
import pathlib as _pathlib
sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parent.parent))

OUT = Path("experiments/outputs")
FIGS = Path("experiments/figures")

#: Read off the published figure; approximate, for visual comparison only.
ANTHROPIC = {
    "rates": [0.0, 0.10, 0.25, 0.50],
    "trust_everyone": [1.00, 0.92, 0.75, 0.475],
    "learns_who_lies": [1.00, 0.92, 0.888, 0.888],
    "models": {
        "Sonnet 4.6": [1.00, 0.92, 0.833, 0.617],
        "Sonnet 5": [1.00, 0.92, 0.812, 0.632],
        "Opus 4.6": [1.00, 0.92, 0.845, 0.721],
        "Opus 4.8": [1.00, 0.92, 0.848, 0.733],
        "Mythos 5": [1.00, 0.92, 0.880, 0.852],
    },
}

COLORS = {
    "sonnet-4-6": "#6BAED6", "sonnet-5": "#08519C",
    "opus-4-6": "#74C476", "opus-4-8": "#238B45",
    "sonnet-4-5-20250929": "#9E9AC8", "opus-5": "#E6550D",
}


def short(model_id: str) -> str:
    return model_id.replace("us.anthropic.claude-", "").replace("-v1:0", "")


def load_ours():
    """(rates, per-model medians, credulous medians, ceiling) or None."""
    path = OUT / "gullibility.json"
    if not path.exists():
        return None
    data = [r for r in json.loads(path.read_text()) if r.get("valid")]
    # Refuse runs where the group-theory table was left in the brief. With
    # that un-poisonable channel available agents bypass the corrupted
    # tools entirely and every curve is flat at the ceiling, which looks
    # like a finding but measures nothing. Filtering on the recorded flag
    # rather than on a filename means a stale file cannot quietly produce
    # a null result.
    usable = [r for r in data if not r.get("theory_in_brief", True)]
    if len(usable) < len(data):
        skipped = len(data) - len(usable)
        print(f"  ignoring {skipped} run(s) with theory_in_brief=True "
              f"(un-poisonable channel; see --no-theory-in-brief)")
    data = usable
    if not data:
        return None
    ceiling = data[0]["bayes_accuracy"]
    analyst = defaultdict(lambda: defaultdict(list))
    credulous = defaultdict(list)
    for r in data:
        rate = r.get("poison_rate", 1.0)
        if r.get("role") == "credulous":
            credulous[rate].append(r["test_accuracy"])
        else:
            analyst[short(r["model_id"])][rate].append(r["test_accuracy"])
    rates = sorted({r.get("poison_rate", 1.0) for r in data})
    return rates, analyst, credulous, ceiling


def main() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ours = load_ours()
    FIGS.mkdir(parents=True, exist_ok=True)
    ncols = 2 if ours else 1
    fig, axes = plt.subplots(1, ncols, figsize=(7.6 * ncols, 5.2))
    axes = np.atleast_1d(axes)
    left = axes[0]

    # ---- Anthropic (digitized) --------------------------------------
    a = ANTHROPIC
    left.fill_between(a["rates"], a["trust_everyone"], a["learns_who_lies"],
                      color="0.85", alpha=0.6, zorder=0)
    left.plot(a["rates"], a["trust_everyone"], ls=":", color="0.35", lw=1.8,
              label="trust everyone")
    left.plot(a["rates"], a["learns_who_lies"], ls="-.", color="0.35", lw=1.8,
              label="learns who lies")
    for name, ys in a["models"].items():
        key = name.lower().replace(" ", "-")
        left.plot(a["rates"], ys, marker="o", mfc="none", ls="--", lw=1.5,
                  color=COLORS.get(key, "#DD8452"), label=name)
    left.set_xlabel("how often the bad source lies")
    left.set_ylabel("routing decision accuracy")
    left.set_title("Anthropic FRT, reported (digitized)\n"
                   "scripted lying scout, scenario ground truth", fontsize=10)
    left.set_ylim(0.45, 1.03)
    left.legend(fontsize=7.5, ncol=2, loc="lower left")
    left.text(0.5, -0.22,
              "open markers / dashed = values read off the published figure,\n"
              "approximate; not our measurements",
              transform=left.transAxes, ha="center", fontsize=8, color="#555")

    # ---- Ours -------------------------------------------------------
    if ours:
        rates, analyst, credulous, ceiling = ours
        right = axes[1]
        cred = [np.median(credulous[r]) if credulous.get(r) else np.nan
                for r in rates]
        oracle = [ceiling] * len(rates)
        if not np.all(np.isnan(cred)):
            right.fill_between(rates, cred, oracle, color="0.85", alpha=0.6,
                               zorder=0)
            right.plot(rates, cred, ls=":", color="0.35", lw=1.8,
                       marker="s", label="credulous agent (measured)")
        right.plot(rates, oracle, ls="-.", color="0.35", lw=1.8,
                   label="provable Bayes ceiling")
        for model, by_rate in analyst.items():
            ys = [np.median(by_rate[r]) if by_rate.get(r) else np.nan
                  for r in rates]
            ns = [len(by_rate.get(r, [])) for r in rates]
            right.plot(rates, ys, marker="o", lw=1.8,
                       color=COLORS.get(model, None),
                       label=f"{model} (n={max(ns)}/pt)")
        right.set_xlabel("fraction of tool calls returning falsified data")
        right.set_ylabel("task accuracy")
        right.set_title("This work\nfalsified tool channel, provable ground truth",
                        fontsize=10)
        right.legend(fontsize=7.5, loc="lower left")
        right.text(0.5, -0.22,
                   "upper bound is the exact Bayes-optimal ceiling, not a\n"
                   "better-behaved policy: the axis is absolutely calibrated",
                   transform=right.transAxes, ha="center", fontsize=8,
                   color="#555")
    else:
        left.text(1.02, 0.5,
                  "our curve pending:\nrun experiments/gullibility_eval.py",
                  transform=left.transAxes, fontsize=9, color="#888",
                  va="center")

    fig.suptitle("Gullibility under corrupted evidence: reported result and "
                 "an exact-ground-truth analogue", fontsize=11.5, y=1.01)
    fig.tight_layout()
    path = FIGS / "fig_gullibility.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")
    if not ours:
        print("(left panel only; our gullibility sweep has not run yet)")


if __name__ == "__main__":
    main()
