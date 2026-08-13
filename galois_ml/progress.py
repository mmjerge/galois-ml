"""tqdm-based progress reporting for long agent runs.

Agent runs take minutes and make dozens of tool calls. With Strands'
callback handler disabled the console stays blank throughout, which is
indistinguishable from a hang (we were already bitten once by mistaking
buffered output for a stalled sweep). These helpers drive a tqdm bar
from the agent's own tool activity.

Two bars are used together:

* an outer bar over the units of work (agents, or conditions), and
* an inner per-agent bar whose postfix shows the running tool count, so
  a long model turn is visibly alive even though the total number of
  tool calls is not known in advance.

The Strands callback payload is an evolving interface, so the handler
ignores anything unexpected rather than aborting an expensive run.
"""

from __future__ import annotations

import sys
import time

from tqdm.auto import tqdm


def outer_bar(total: int, desc: str):
    """Progress bar over units of work (e.g. agents or conditions)."""
    return tqdm(
        total=total,
        desc=desc,
        unit="run",
        position=0,
        leave=True,
        file=sys.stdout,
        dynamic_ncols=True,
    )


class AgentProgress:
    """Per-agent tqdm bar advanced by Strands tool-use callbacks.

    The total tool count is unknown ahead of time, so the bar tracks
    tool calls without a total (tqdm renders elapsed time and a rate),
    and the postfix carries the most recent tool name.

    Usage:
        with AgentProgress("agent 2/5") as prog:
            agent = Agent(..., callback_handler=prog.handler)
            result = agent(prompt)
        prog.tool_calls  # observed count
    """

    def __init__(self, label: str, position: int = 1):
        self.label = label
        self.start = time.monotonic()
        self.tool_calls = 0
        self._seen: set[str] = set()
        self._bar = tqdm(
            desc=f"  {label}",
            unit="tool",
            position=position,
            leave=False,
            file=sys.stdout,
            dynamic_ncols=True,
            # note: tqdm prefixes {postfix} with ", " when set via
            # set_postfix_str, so the tool name is carried in {desc}
            # instead to keep the line clean
            bar_format="{desc}: {n} tool calls [{elapsed}]",
        )

    @property
    def elapsed(self) -> float:
        return time.monotonic() - self.start

    def handler(self, **kwargs) -> None:
        """Passed to Strands as `callback_handler`."""
        try:
            tool = kwargs.get("current_tool_use") or {}
            if not isinstance(tool, dict):
                return
            name = tool.get("name")
            tool_id = tool.get("toolUseId") or tool.get("tool_use_id") or name
            # the callback fires repeatedly while a tool call streams in;
            # count each distinct invocation once
            if name and tool_id and tool_id not in self._seen:
                self._seen.add(tool_id)
                self.tool_calls += 1
                self._bar.set_description_str(
                    f"  {self.label} -> {name}", refresh=False
                )
                self._bar.update(1)
        except Exception:
            pass  # never let reporting break a run

    def close(self, outcome: str = "") -> None:
        if outcome:
            self._bar.set_description_str(
                f"  {self.label} ({outcome})", refresh=True
            )
        self._bar.close()

    def __enter__(self) -> "AgentProgress":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close("failed" if exc_type is not None else "")


def write(message: str) -> None:
    """Emit a line without corrupting active tqdm bars."""
    tqdm.write(message, file=sys.stdout)
