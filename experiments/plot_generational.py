"""WITHDRAWN: cross-study generational overlay.

This script produced a chart that could not support the reading it
invited, so it has been withdrawn rather than left in place. Recorded
here because the defects are instructive.

1. FALSE ALIGNMENT. Two studies were plotted against one x-axis, but the
   series had different lengths (5 published generations vs 6 measured
   here) and the ticks were labelled with our model names. At tick 0 the
   published marker was Sonnet 4.6 while ours was Sonnet 4.5, so the two
   lines were not aligned to the same models and point-by-point comparison
   was meaningless. A claim that "the endpoints agree" was drawn from a
   chart whose endpoints are different models.

2. NO REAL ORDERING. The published legend order is not chronological
   (Sonnet 5 precedes Opus 4.6), so "generation" was not a well-defined
   axis in either series.

3. UNINTERPRETABLE NORMALIZATION. Expressing outcomes as "fraction of the
   naive-to-oracle gap recovered" goes negative when a run falls below the
   naive floor, which is exactly what label inversion does. A value of
   -0.89 is not a fraction of anything. Inversion is a distinct failure
   mode, not a degree of recovery, and belongs on a categorical axis.

4. MIXED SCALES. The second panel put accuracy lost (0 to 0.37) and a
   detection rate (0 to 1) on a shared y-axis, so "detection is exactly
   zero in every run" read as merely "detection is small".

The generational results are reported as tables in the README instead,
where each cell states its own model, metric, and n. The one cross-study
figure retained is experiments/plot_gullibility.py, which is legitimately
comparable: a shared x-axis (corruption rate), the same measured quantity
(task accuracy), and explicit naive and oracle baselines in both studies.
"""

import sys

print(__doc__, file=sys.stderr)
raise SystemExit(
    "withdrawn: see the module docstring, and use experiments/report.py "
    "for the generational tables"
)
