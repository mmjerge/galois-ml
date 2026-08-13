"""Fast, actionable dependency checks for the agent experiments.

The agent experiments need optional dependencies (strands-agents, boto3)
that are commonly installed into one interpreter while the scripts get
launched with another. Failing at the first Bedrock call, or worse once
per agent inside a loop, wastes time and obscures the cause. These
helpers fail immediately and name a working interpreter if one exists.
"""

from __future__ import annotations

import importlib.util
import shutil
import subprocess
import sys

#: interpreters worth probing for a working environment, in order
_CANDIDATES = (
    "/opt/anaconda3/bin/python3",
    "/opt/homebrew/bin/python3",
    "/usr/local/bin/python3",
    "/usr/bin/python3",
)


def _has(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


def _find_interpreter_with(modules: tuple[str, ...]) -> str | None:
    """A python executable (other than the current one) that imports all
    the given modules."""
    probe = "import importlib.util as u; " + "; ".join(
        f"assert u.find_spec({m!r})" for m in modules
    )
    for candidate in _CANDIDATES:
        if candidate == sys.executable or not shutil.which(candidate):
            continue
        try:
            done = subprocess.run(
                [candidate, "-c", probe],
                capture_output=True, timeout=30,
            )
        except (OSError, subprocess.SubprocessError):
            continue
        if done.returncode == 0:
            return candidate
    return None


def require(*modules: str, extra: str | None = None) -> None:
    """Exit with an actionable message unless every module is importable.

    Args:
        modules: import names that must be available.
        extra: the optional-dependency group that provides them, used in
            the suggested pip command.
    """
    missing = [m for m in modules if not _has(m)]
    if not missing:
        return

    lines = [
        "",
        f"Missing required package(s): {', '.join(missing)}",
        f"Current interpreter: {sys.executable}",
    ]
    working = _find_interpreter_with(modules)
    if working:
        argv = " ".join(sys.argv)
        lines += [
            "",
            "These packages ARE available in another interpreter. Rerun with:",
            f"  {working} {argv}",
        ]
    else:
        target = f'".[{extra}]"' if extra else "."
        lines += [
            "",
            "Install them into this interpreter:",
            f"  {sys.executable} -m pip install -e {target}",
        ]
    lines.append("")
    print("\n".join(lines), file=sys.stderr)
    raise SystemExit(2)


def require_agents() -> None:
    """Dependencies for the Strands-based multi-agent experiments."""
    require("strands", "boto3", extra="agents,bedrock")
    scrub_stale_bedrock_token()


def require_bedrock() -> None:
    """Dependencies for the single-model Bedrock harness."""
    require("boto3", extra="bedrock")
    scrub_stale_bedrock_token()


def scrub_stale_bedrock_token() -> None:
    """Remove AWS_BEARER_TOKEN_BEDROCK from the environment.

    boto3 gives this token precedence over ordinary AWS credentials for
    Bedrock calls, so a stale one (commonly a hardcoded export in a shell
    profile) produces "Bearer Token has expired" no matter how recently
    the session was authenticated. Since it masks working credentials, we
    drop it and say so rather than failing per-agent deep inside a run.

    Set GALOIS_ML_KEEP_BEDROCK_TOKEN=1 to retain it (e.g. if it is the
    credential you actually intend to use).
    """
    import os

    if os.environ.get("GALOIS_ML_KEEP_BEDROCK_TOKEN") == "1":
        return
    if os.environ.pop("AWS_BEARER_TOKEN_BEDROCK", None) is not None:
        print(
            "note: unset AWS_BEARER_TOKEN_BEDROCK for this run; it takes\n"
            "      precedence over your session credentials and is a common\n"
            "      source of 'Bearer Token has expired'. Set\n"
            "      GALOIS_ML_KEEP_BEDROCK_TOKEN=1 to keep it.",
            file=sys.stderr,
        )
