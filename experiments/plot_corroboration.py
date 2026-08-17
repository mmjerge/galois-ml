"""WITHDRAWN: cross-study aggregation-failure overlay.

Withdrawn for the same reason as plot_generational.py: it placed a
published result and ours side by side in a way that invited a
quantitative comparison the data cannot support.

The specific problem here was that the two results share a theme but not a
mechanism. The published hidden-profile experiment distributes *different*
private facts across agents, and the failure is not surfacing or trusting
them. Our agents all see *identical* data, and the failure is correlated
errors on the same items. Both show group aggregation underdelivering
against its ceiling, but the quantities are not commensurable, and putting
them in adjacent panels with similar bar styling implied they were.

The conformity result stands on its own and is reported as a table in the
README:

  5 agents per task, temperature 1.0
  sextic    5/5 distinct conjectures, error correlation 14.5x
            (21.6x on avoidable errors), 1.25 of 5 effective independent
  degree21  5/5 distinct conjectures, error correlation 25.5x
            (112.6x on avoidable errors), 1.05 of 5 effective independent

A single-study figure for this may be worth building later. It does not
need a published result beside it to make its point.
"""

import sys

print(__doc__, file=sys.stderr)
raise SystemExit(
    "withdrawn: see the module docstring, and use experiments/report.py "
    "for the conformity tables"
)
