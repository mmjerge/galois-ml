"""Pool sharded sequential-gullibility results and report with power.

Decisions are pooled across episodes within each cell rather than
averaging per-episode accuracies: per-episode accuracy is quantized to
1/n_decisions, so averaging episode means discards resolution and
understates precision. Confidence intervals are Wilson rather than
normal-approximation, because accuracies here sit near 1.0 where the
normal interval misbehaves (it can exceed 1 or give zero width at p=1).

The rate-0 control is checked first and reported separately. A cell whose
honest control is far below the ceiling cannot support any claim about
gullibility: the agent is failing the task, not misplacing trust. Two of
our earlier runs were invalidated exactly this way.

Run:  python experiments/report_sequential.py
"""

from __future__ import annotations

import glob
import json
import math
import pathlib as _pathlib
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parent.parent))

OUT = Path("experiments/outputs")
#: a control this far below the ceiling invalidates the cell's curve
CONTROL_TOLERANCE = 0.15


def wilson(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval: behaves near p=0 and p=1, unlike the normal
    approximation, which matters because most cells here sit near 1.0."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def load() -> list[dict]:
    records = []
    for path in sorted(glob.glob(str(OUT / "rep_*.json"))) + \
                sorted(glob.glob(str(OUT / "sequential_gullibility.json"))):
        try:
            records.extend(
                r for r in json.loads(Path(path).read_text())
                if r.get("valid")
            )
        except (json.JSONDecodeError, OSError):
            continue  # mid-write
    return records


def main() -> None:
    data = load()
    if not data:
        print("no sequential results found")
        return

    cells = defaultdict(lambda: {"c": 0, "n": 0, "attr": [], "eps": 0})
    for r in data:
        key = (r["model_id"].replace("us.anthropic.claude-", ""),
               r["role"], r["lie_rate"])
        cell = cells[key]
        cell["c"] += r["n_correct"]
        cell["n"] += r["n_decisions"]
        cell["attr"].append(bool(r["correct_attribution"]))
        cell["eps"] += 1

    ceiling = max((r.get("bayes_accuracy") or 0) for r in data)
    task = data[0]["task"]
    print(f"task {task}, ceiling {ceiling:.4f}, "
          f"{sum(c['eps'] for c in cells.values())} episodes, "
          f"{sum(c['n'] for c in cells.values())} scored decisions\n")

    # control check per (model, role)
    print("control check (lie rate 0.0 must be near the ceiling)")
    valid_curves = set()
    for (model, role, rate), c in sorted(cells.items()):
        if rate != 0.0:
            continue
        acc = c["c"] / c["n"] if c["n"] else float("nan")
        ok = acc >= ceiling - CONTROL_TOLERANCE
        valid_curves.add((model, role)) if ok else None
        lo, hi = wilson(c["c"], c["n"])
        print(f"  {model:<13}{role:<11}acc {acc:.3f} [{lo:.2f},{hi:.2f}] "
              f"n={c['n']:<5}{'OK' if ok else 'FAILS -- curve not interpretable'}")

    print(f"\n{'model':<13}{'role':<11}{'lie':>5}{'acc':>8}{'95% Wilson':>16}"
          f"{'n':>6}{'attrib':>8}{'valid':>7}")
    for (model, role, rate), c in sorted(cells.items()):
        acc = c["c"] / c["n"] if c["n"] else float("nan")
        lo, hi = wilson(c["c"], c["n"])
        attr = sum(c["attr"]) / len(c["attr"]) if c["attr"] else float("nan")
        flag = "yes" if (model, role) in valid_curves else "NO"
        print(f"{model:<13}{role:<11}{rate:>5}{acc:>8.3f}"
              f"{f'[{lo:.3f},{hi:.3f}]':>16}{c['n']:>6}{attr:>7.0%}{flag:>7}")

    # gullibility gap: listener vs credulous at matched rate
    print("\ngullibility gap (credulous minus listener; positive means "
          "instructed trust cost accuracy)")
    models = sorted({m for m, _, _ in cells})
    rates = sorted({r for _, _, r in cells})
    for model in models:
        row = []
        for rate in rates:
            lis = cells.get((model, "listener", rate))
            cre = cells.get((model, "credulous", rate))
            if lis and cre and lis["n"] and cre["n"]:
                row.append(f"{rate}: {lis['c']/lis['n'] - cre['c']/cre['n']:+.3f}")
        if row:
            print(f"  {model:<13}" + "   ".join(row))

    print("\nreading notes")
    print("* pooled over decisions, not averaged over episodes: episode")
    print("  accuracy is quantized to 1/n_decisions and averaging loses")
    print("  resolution.")
    print("* Wilson intervals, since cells sit near 1.0 where the normal")
    print("  approximation can exceed 1 or collapse to zero width.")
    print("* a curve whose rate-0 control fails is marked NO and should not")
    print("  be read as a gullibility measurement.")


if __name__ == "__main__":
    main()
