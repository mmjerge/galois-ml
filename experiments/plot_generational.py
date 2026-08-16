"""Generational trend: Anthropic's reported curve vs ours, normalized.

Their gullibility figure and our cross-model sweep measure different
things on different axes, so a naive side-by-side would be misleading.
Both, however, admit the same normalization: how much of the gap between
a naive baseline and an oracle does a given model recover?

  Anthropic (at lie rate 0.50, the most discriminating point):
      recovered = (model - trust_everyone) / (learns_who_lies - trust_everyone)
  This work (channel `combined`, the analogue with no honest channel):
      recovered = (model - floor) / (own clean control - floor)
  where floor is the majority-class baseline for the task, and the upper
  bound is that model's OWN clean-tool accuracy, which controls for
  differences in raw task competence between generations.

The second panel shows what the normalization hides and what we can
measure but they cannot separate: harm and detection are independent.
Newer models take no damage from a provably false channel while never
once reporting it.

Their values are digitized from the published figure and are approximate;
they are drawn with open markers and labelled as such.

Source: "Patterns and problems in emerging multiagent systems", Anthropic
Frontier Red Team, 2026-08-13.
https://www.anthropic.com/research/multiagent-systems

Run:  python experiments/plot_generational.py
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

import pathlib as _pathlib
sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parent.parent))

OUT = Path("experiments/outputs")
FIGS = Path("experiments/figures")

#: Digitized from the published figure at lie rate 0.50 (approximate).
ANTHROPIC_AT_HALF = {
    "trust_everyone": 0.475,
    "learns_who_lies": 0.888,
    "models": {          # ordered oldest -> newest as presented
        "Sonnet 4.6": 0.617,
        "Sonnet 5": 0.632,
        "Opus 4.6": 0.721,
        "Opus 4.8": 0.733,
        "Mythos 5": 0.852,
    },
}

#: Our generations, oldest to newest, with display labels matching theirs
#: where the generation corresponds.
OUR_ORDER = [
    ("sonnet-4-5-20250929-v1:0", "Sonnet 4.5"),
    ("sonnet-4-6", "Sonnet 4.6"),
    ("opus-4-6-v1", "Opus 4.6"),
    ("opus-4-8", "Opus 4.8"),
    ("sonnet-5", "Sonnet 5"),
    ("opus-5", "Opus 5"),
]

#: Balanced binary tasks: chance is 0.5, which is the naive floor here.
TASK_FLOOR = 0.5


def load_ours(attack: str = "combined"):
    path = OUT / "cross_model.json"
    if not path.exists():
        return None
    data = [r for r in json.loads(path.read_text()) if r.get("valid")]
    if not data:
        return None
    by = defaultdict(lambda: defaultdict(list))
    for r in data:
        key = r["model_id"].replace("us.anthropic.claude-", "")
        by[key][r["attack"]].append(r)
    rows = []
    for key, label in OUR_ORDER:
        atk, ctrl = by.get(key, {}).get(attack, []), by.get(key, {}).get("none", [])
        if not atk or not ctrl:
            continue
        a = float(np.median([r["test_accuracy"] for r in atk]))
        c = float(np.median([r["test_accuracy"] for r in ctrl]))
        span = c - TASK_FLOOR
        rows.append({
            "label": label,
            "attacked": a,
            "control": c,
            # fraction of its own achievable range that survives the attack
            "recovered": (a - TASK_FLOOR) / span if span > 0 else np.nan,
            "detect_attack": float(np.mean(
                [bool(r["attack_surfaced"]) for r in atk])),
            "detect_impossible": float(np.mean(
                [bool(r["attack_surfaced"])
                 for r in by[key].get("impossible", [])] or [np.nan])),
            "harm_impossible": (
                float(np.median([r["test_accuracy"]
                                 for r in by[key]["impossible"]]) - c)
                if by[key].get("impossible") else np.nan),
            "n": len(atk),
        })
    return rows


def main() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ours = load_ours("combined")
    if not ours:
        print("no cross-model results yet; run experiments/cross_model_eval.py")
        return
    FIGS.mkdir(parents=True, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.5, 5.4))

    # ---- Panel 1: normalized gap recovery, both studies ------------
    a = ANTHROPIC_AT_HALF
    span = a["learns_who_lies"] - a["trust_everyone"]
    their_labels = list(a["models"])
    their_rec = [(v - a["trust_everyone"]) / span for v in a["models"].values()]

    our_labels = [r["label"] for r in ours]
    our_rec = [r["recovered"] for r in ours]

    x1 = np.arange(len(their_labels))
    x2 = np.arange(len(our_labels))
    ax1.plot(x1, their_rec, "o--", mfc="none", color="#DD8452", lw=1.8,
             label="Anthropic FRT (digitized, lie rate 0.50)")
    ax1.plot(x2 + 0.0, our_rec, "s-", color="#4C72B0", lw=2.0,
             label="this work (channel `combined`)")
    ax1.axhline(1.0, ls="-.", color="0.4", lw=1.2)
    ax1.text(0.02, 1.02, "oracle / own clean control", fontsize=8, color="0.35")
    ax1.axhline(0.0, ls=":", color="0.4", lw=1.2)
    ax1.text(0.02, 0.03, "naive floor", fontsize=8, color="0.35")
    all_labels = sorted(set(their_labels) | set(our_labels),
                        key=lambda s: (our_labels + their_labels).index(s)
                        if s in our_labels else 99)
    ax1.set_xticks(range(max(len(their_labels), len(our_labels))))
    ax1.set_xticklabels(
        our_labels if len(our_labels) >= len(their_labels) else their_labels,
        rotation=20, ha="right", fontsize=9,
    )
    ax1.set_ylabel("fraction of naive-to-oracle gap recovered")
    ax1.set_ylim(-0.15, 1.2)
    ax1.set_title("Generational trend, normalized so the two studies are\n"
                  "commensurable (higher = more resistant to corrupted evidence)",
                  fontsize=10)
    ax1.legend(fontsize=8, loc="lower right")
    ax1.text(0.5, -0.30,
             "x-axis is generation order within each study, not a shared scale;\n"
             "labels shown are ours where the two overlap",
             transform=ax1.transAxes, ha="center", fontsize=8, color="#555")

    # ---- Panel 2: harm and detection are independent ----------------
    labels = [r["label"] for r in ours]
    xs = np.arange(len(labels))
    harm = [-r["harm_impossible"] for r in ours]        # positive = damage
    det = [r["detect_impossible"] for r in ours]
    ax2.bar(xs - 0.2, harm, 0.4, label="accuracy lost to `impossible`",
            color="#C44E52")
    ax2.bar(xs + 0.2, det, 0.4, label="detection rate on `impossible`",
            color="#55A868")
    ax2.set_xticks(xs, labels, rotation=20, ha="right", fontsize=9)
    ax2.set_ylim(-0.03, max(0.45, max(harm) * 1.25))
    ax2.set_ylabel("accuracy lost  /  detection rate")
    ax2.set_title("What the normalization hides:\n"
                  "harm falls to zero, detection never leaves zero",
                  fontsize=10)
    ax2.legend(fontsize=8)
    for i, (h, dv) in enumerate(zip(harm, det)):
        if h > 0.01:
            ax2.text(i - 0.2, h + 0.012, f"{h:.2f}", ha="center", fontsize=8)
    ax2.text(0.5, -0.30,
             "the provable lie is never reported by ANY generation, even those\n"
             "it no longer harms: robustness improved, vigilance did not",
             transform=ax2.transAxes, ha="center", fontsize=8, color="#555")

    fig.suptitle("Do newer models resist corrupted evidence? "
                 "Reported trend and an exact-ground-truth analogue",
                 fontsize=11.5, y=1.02)
    fig.tight_layout()
    path = FIGS / "fig_generational.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}\n")

    print(f"{'model':<14}{'combined acc':>13}{'own clean':>11}"
          f"{'gap recovered':>15}{'impossible harm':>17}{'detect imp.':>12}")
    for r in ours:
        print(f"{r['label']:<14}{r['attacked']:>13.3f}{r['control']:>11.3f}"
              f"{r['recovered']:>15.2f}{r['harm_impossible']:>+17.3f}"
              f"{r['detect_impossible']:>12.0%}")
    print("\ntheir normalized values at lie rate 0.50 (digitized):")
    for k, v in zip(their_labels, their_rec):
        print(f"  {k:<12}{v:>6.2f}")


if __name__ == "__main__":
    main()
