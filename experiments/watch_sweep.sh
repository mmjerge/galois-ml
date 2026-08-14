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
PIDFILE="$REPO/experiments/outputs/sweep.pid"

# Do not hardcode an interpreter: this runs both on macOS (anaconda) and
# on EC2 (system python3).
for _candidate in "${GALOIS_PYTHON:-}" /opt/anaconda3/bin/python3 python3 python; do
  [[ -z "$_candidate" ]] && continue
  if command -v "$_candidate" >/dev/null 2>&1; then PYTHON="$_candidate"; break; fi
done
PYTHON="${PYTHON:-python3}"

render() {
  cd "$REPO"
  # the sweep may be running under a pidfile (run_sweep_here.sh) or
  # inside tmux (run_all.sh / the EC2 host), so check both
  if [[ -f "$PIDFILE" ]] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
    echo "sweep RUNNING (pid $(cat "$PIDFILE"))"
  elif command -v tmux >/dev/null 2>&1 && tmux has-session -t galois 2>/dev/null; then
    echo "sweep RUNNING (tmux session 'galois')"
  elif pgrep -f "_sweep.sh" >/dev/null 2>&1; then
    echo "sweep RUNNING (pid $(pgrep -f '_sweep.sh' | head -1))"
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
