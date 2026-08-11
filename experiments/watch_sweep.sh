#!/usr/bin/env bash
# Show live sweep progress, reading the incrementally-written result
# files rather than the log.
#
# Useful because Python block-buffers stdout to a redirected log, so the
# log can look frozen while the sweep is in fact progressing. The JSON
# result files are flushed after every run, so they are the reliable
# progress signal.
#
# Usage:  experiments/watch_sweep.sh [--once]

set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="/opt/anaconda3/bin/python3"
PIDFILE="$REPO/experiments/outputs/sweep.pid"

render() {
  cd "$REPO"
  if [[ -f "$PIDFILE" ]] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
    echo "sweep RUNNING (pid $(cat "$PIDFILE"))"
  else
    echo "sweep not running (finished, or never started)"
  fi
  echo
  "$PYTHON" - <<'PY'
import glob, json, os

for path in sorted(glob.glob("experiments/outputs/tool_poisoning_*.json")
                   + glob.glob("experiments/outputs/multi_agent_*.json")):
    try:
        records = json.load(open(path))
    except (json.JSONDecodeError, OSError):
        continue  # mid-write
    if not records:
        continue
    print(f"{os.path.basename(path)}  ({len(records)} runs)")
    for r in records:
        acc = r.get("test_accuracy")
        acc_s = f"{acc:.4f}" if acc is not None else "FAIL"
        reg = r.get("regret")
        reg_s = f"{reg:+.4f}" if reg is not None else "  --  "
        cond = r.get("attack") or r.get("condition") or "?"
        critic = r.get("with_critic")
        critic_s = "+critic" if critic else ""
        seed = f"s{r['seed']}" if "seed" in r else ""
        flag = ""
        if r.get("attack_surfaced") is True:
            flag = " FLAGGED"
        print(f"  {r.get('task','?'):<10} {cond+critic_s:<22} {seed:<3} "
              f"acc {acc_s}  regret {reg_s}{flag}")
    print()
PY
}

if [[ "${1:-}" == "--once" ]]; then
  render
else
  while true; do
    clear
    render
    echo "(refreshing every 20s; Ctrl-C to stop watching -- the sweep keeps running)"
    sleep 20
  done
fi
