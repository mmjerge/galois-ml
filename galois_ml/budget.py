"""Tool-call budgets for agent runs.

Agents on these tasks can iterate indefinitely: the optimal rule for
several tasks is statistical, so validation accuracy never reaches 1.0
and an agent tuning thresholds always sees room to improve. Observed
runs land around 50 tool calls, but there is nothing in the loop that
forces termination.

A budget bounds cost and wall time. The design point that matters: when
the budget is exhausted the tools do NOT raise. Raising would abort the
run and discard the agent's work, turning a cost control into a lost
sample. Instead each tool returns a message instructing the agent to
finalize its answer immediately, so the run still yields a scoreable
rule and conjecture. Budget exhaustion is recorded so results can note
which runs were truncated.
"""

from __future__ import annotations

from dataclasses import dataclass, field

#: Default cap. Roughly twice the observed usage of a run that
#: terminated on its own, so it bounds pathological loops without
#: truncating normal work.
DEFAULT_MAX_TOOL_CALLS = 100

_EXHAUSTED_MESSAGE = (
    "TOOL BUDGET EXHAUSTED ({used}/{limit} calls used). No further tool "
    "calls will be answered. Stop investigating now and immediately "
    "return your final JSON object with the best conjecture and "
    "predict(zeta) implementation you have arrived at so far."
)


@dataclass
class ToolBudget:
    """Counts tool calls against a limit.

    A limit of None or <= 0 means unlimited.
    """

    limit: int | None = DEFAULT_MAX_TOOL_CALLS
    used: int = 0
    exhausted_hits: int = 0
    per_tool: dict = field(default_factory=dict)

    @property
    def unlimited(self) -> bool:
        return self.limit is None or self.limit <= 0

    @property
    def exhausted(self) -> bool:
        return not self.unlimited and self.used >= self.limit

    def charge(self, tool_name: str) -> str | None:
        """Record one call. Returns None if the call may proceed, or the
        message the tool should return instead if the budget is spent."""
        if self.unlimited:
            self.used += 1
            self.per_tool[tool_name] = self.per_tool.get(tool_name, 0) + 1
            return None
        if self.used >= self.limit:
            self.exhausted_hits += 1
            return _EXHAUSTED_MESSAGE.format(used=self.used, limit=self.limit)
        self.used += 1
        self.per_tool[tool_name] = self.per_tool.get(tool_name, 0) + 1
        return None

    def summary(self) -> dict:
        return {
            "limit": self.limit,
            "used": self.used,
            "exhausted": self.exhausted,
            "refused_calls": self.exhausted_hits,
            "per_tool": dict(self.per_tool),
        }
