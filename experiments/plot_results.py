"""Figures and summary statistics for the benchmark experiments.

Reads the JSON result files in experiments/outputs/ and produces:

  fig_ceiling_vs_models.png   accuracy vs the Bayes ceiling per task
                              (regret framing: the ceiling is the bar to
                              beat, not 1.0)
  fig_attack_harm.png         regret by attack channel, with and without
                              a critic, mean +/- range over seeds
  fig_detection.png           attack detection rate by channel
  fig_contamination.png       published vs held-out tasks under the
                              derivation protocol (contamination probe)

Also writes summary.json with per-condition aggregates (mean, std, n),
which is what belongs in a paper table.

Run:  python experiments/plot_results.py [--outputs DIR]
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

OUT = Path("experiments/outputs")
FIGS = Path("experiments/figures")

ATTACK_ORDER = ["none", "data", "verification", "injection", "combined"]


def _load(pattern: str) -> list[dict]:
    records = []
    for path in sorted(OUT.glob(pattern)):
        try:
            payload = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
        records.extend(payload if isinstance(payload, list)
                       else payload.get("results", []))
    return records


def _agg(values: list[float]) -> dict:
    arr = np.array([v for v in values if v is not None], dtype=float)
    if arr.size == 0:
        return {"mean": None, "std": None, "n": 0}
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
        "n": int(arr.size),
    }


def plot_attack_harm(records: list[dict], summary: dict) -> None:
    """Regret by attack channel, split by critic presence."""
    import matplotlib.pyplot as plt

    valid = [r for r in records if r.get("valid") and r.get("regret") is not None]
    if not valid:
        return
    groups: dict[tuple, list[float]] = defaultdict(list)
    for r in valid:
        groups[(r["attack"], bool(r.get("with_critic")))].append(r["regret"])

    attacks = [a for a in ATTACK_ORDER if any(k[0] == a for k in groups)]
    x = np.arange(len(attacks))
    width = 0.38
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    for offset, critic, label, color in (
        (-width / 2, False, "analyst alone", "#4C72B0"),
        (width / 2, True, "analyst + critic", "#DD8452"),
    ):
        means, errs = [], []
        for a in attacks:
            vals = groups.get((a, critic), [])
            means.append(np.mean(vals) if vals else np.nan)
            errs.append(
                (np.std(vals, ddof=1) if len(vals) > 1 else 0.0) if vals else 0.0
            )
        ax.bar(x + offset, means, width, yerr=errs, capsize=3,
               label=label, color=color)
        summary.setdefault("attack_harm", {})[
            "critic" if critic else "no_critic"
        ] = {a: _agg(groups.get((a, critic), [])) for a in attacks}

    ax.set_xticks(x, attacks)
    ax.set_xlabel("poisoned tool channel")
    ax.set_ylabel("regret vs Bayes ceiling  (lower is better)")
    ax.set_title("Harm from tool-channel poisoning")
    ax.axhline(0, color="black", lw=0.8)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGS / "fig_attack_harm.png", dpi=200)
    plt.close(fig)


def plot_detection(records: list[dict], summary: dict) -> None:
    """Fraction of runs in which an agent flagged the anomaly."""
    import matplotlib.pyplot as plt

    valid = [
        r for r in records
        if r.get("valid") and r.get("attack") not in (None, "none")
        and r.get("attack_surfaced") is not None
    ]
    if not valid:
        return
    by_attack: dict[str, list[bool]] = defaultdict(list)
    for r in valid:
        by_attack[r["attack"]].append(bool(r["attack_surfaced"]))
    attacks = [a for a in ATTACK_ORDER if a in by_attack]
    rates = [float(np.mean(by_attack[a])) for a in attacks]

    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.bar(attacks, rates, color="#55A868")
    for i, (a, rate) in enumerate(zip(attacks, rates)):
        ax.text(i, rate + 0.02, f"{rate:.0%}\n(n={len(by_attack[a])})",
                ha="center", fontsize=9)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("runs where the attack was flagged")
    ax.set_xlabel("poisoned tool channel")
    ax.set_title("Attack detection rate (generous keyword heuristic)")
    fig.tight_layout()
    fig.savefig(FIGS / "fig_detection.png", dpi=200)
    plt.close(fig)
    summary["detection_rate"] = {
        a: {"rate": float(np.mean(by_attack[a])), "n": len(by_attack[a])}
        for a in attacks
    }


def plot_ceiling(records: list[dict], summary: dict) -> None:
    """Accuracy against the Bayes ceiling per task, across sources."""
    import matplotlib.pyplot as plt

    valid = [
        r for r in records
        if r.get("valid") and r.get("test_accuracy") is not None
        and r.get("bayes_accuracy")
    ]
    if not valid:
        return
    by_task: dict[str, list[float]] = defaultdict(list)
    ceilings: dict[str, float] = {}
    for r in valid:
        by_task[r["task"]].append(r["test_accuracy"])
        ceilings[r["task"]] = r["bayes_accuracy"]
    tasks = sorted(by_task)
    x = np.arange(len(tasks))

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    means = [np.mean(by_task[t]) for t in tasks]
    errs = [
        np.std(by_task[t], ddof=1) if len(by_task[t]) > 1 else 0.0
        for t in tasks
    ]
    ax.bar(x, means, 0.55, yerr=errs, capsize=3, color="#4C72B0",
           label="agent accuracy")
    for i, t in enumerate(tasks):
        ax.hlines(ceilings[t], i - 0.32, i + 0.32, color="crimson", lw=2.2,
                  label="Bayes ceiling" if i == 0 else None)
    ax.set_xticks(x, tasks, rotation=15)
    ax.set_ylabel("test accuracy")
    ax.set_title("Agent accuracy vs the provable ceiling")
    ax.set_ylim(0.4, 1.02)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGS / "fig_ceiling_vs_models.png", dpi=200)
    plt.close(fig)
    summary["accuracy_vs_ceiling"] = {
        t: {"accuracy": _agg(by_task[t]), "ceiling": ceilings[t]}
        for t in tasks
    }


def plot_contamination(summary: dict) -> None:
    """Published vs held-out tasks under each protocol (LLM harness)."""
    import matplotlib.pyplot as plt

    from galois_ml.groups import PAPER_TASKS

    records = [
        r for r in _load("llm_eval_*.json")
        if r.get("valid_response") and r.get("regret") is not None
    ]
    if not records:
        return
    buckets: dict[tuple, list[float]] = defaultdict(list)
    for r in records:
        held = r["task"] not in PAPER_TASKS
        buckets[(r["protocol"], held)].append(r["regret"])
    protocols = sorted({k[0] for k in buckets})
    x = np.arange(len(protocols))
    width = 0.38

    fig, ax = plt.subplots(figsize=(7, 4.2))
    for offset, held, label, color in (
        (-width / 2, False, "published tasks", "#8172B3"),
        (width / 2, True, "held-out tasks", "#937860"),
    ):
        means = [np.mean(buckets.get((p, held), [np.nan])) for p in protocols]
        errs = [
            (np.std(buckets[(p, held)], ddof=1)
             if len(buckets.get((p, held), [])) > 1 else 0.0)
            for p in protocols
        ]
        ax.bar(x + offset, means, width, yerr=errs, capsize=3,
               label=label, color=color)
    ax.set_xticks(x, [p.replace("_", " ") for p in protocols])
    ax.set_ylabel("regret vs Bayes ceiling")
    ax.set_title("Contamination probe: published vs held-out criteria")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGS / "fig_contamination.png", dpi=200)
    plt.close(fig)
    summary["contamination"] = {
        f"{protocol}/{'held_out' if held else 'published'}": _agg(values)
        for (protocol, held), values in buckets.items()
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outputs", default=str(OUT))
    args = ap.parse_args()
    globals()["OUT"] = Path(args.outputs)
    FIGS.mkdir(parents=True, exist_ok=True)

    summary: dict = {}
    poisoning = _load("tool_poisoning_*.json")
    agents = _load("multi_agent_*.json")

    plot_attack_harm(poisoning, summary)
    plot_detection(poisoning, summary)
    plot_ceiling(poisoning + agents, summary)
    plot_contamination(summary)

    (OUT / "summary.json").write_text(json.dumps(summary, indent=1))
    figs = sorted(p.name for p in FIGS.glob("*.png"))
    print("figures:", ", ".join(figs) if figs else "(none; no data yet)")
    print(f"summary -> {OUT / 'summary.json'}")
    for section, payload in summary.items():
        print(f"\n[{section}]")
        print(json.dumps(payload, indent=1)[:900])


if __name__ == "__main__":
    main()
