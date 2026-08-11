#!/usr/bin/env bash
# Run the sweep from THIS shell, in the background, surviving terminal
# close. Use this when authentication is session-scoped: authenticate
# normally in your terminal, then run this script. The background job
# inherits this shell's environment, so it inherits your credentials.
#
# This avoids the tmux pitfall where the tmux *server* caches a stale
# environment from whenever it first started (kill-server, not just
# kill-session, is needed to reset it).
#
# Usage:
#   <authenticate in this terminal first>
#   experiments/run_sweep_here.sh [MODEL_ID] [SEEDS]
#
# Monitor:  tail -f experiments/outputs/run_all.log
# Stop:     kill <PID printed below>   (or: pkill -f _sweep.sh)

set -euo pipefail

MODEL_ID="${1:-us.anthropic.claude-sonnet-4-5-20250929-v1:0}"
SEEDS="${2:-3}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG="$REPO/experiments/outputs/run_all.log"
PIDFILE="$REPO/experiments/outputs/sweep.pid"
PYTHON="/opt/anaconda3/bin/python3"

cd "$REPO"
mkdir -p experiments/outputs
chmod +x experiments/_sweep.sh

if [[ -f "$PIDFILE" ]] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
  echo "a sweep is already running (PID $(cat "$PIDFILE"))."
  echo "  monitor: tail -f $LOG"
  echo "  stop:    kill $(cat "$PIDFILE")"
  exit 1
fi

# Fail fast, in the foreground, while the user is still watching.
echo "verifying Bedrock access from this shell..."
if ! "$PYTHON" -c "
from galois_ml.llm_eval import BedrockModel
m = BedrockModel('$MODEL_ID', max_tokens=64)
print('  bedrock ok:', m('Reply with the single word: ready').strip()[:40])
"; then
  cat <<EOF

Bedrock is not reachable from this shell, so the sweep was NOT started.
Authenticate in THIS terminal, then rerun this script. To check by hand:
  $PYTHON -c "from galois_ml.llm_eval import BedrockModel as B; print(B('$MODEL_ID', max_tokens=32)('say ready'))"
EOF
  exit 1
fi

nohup bash experiments/_sweep.sh "$MODEL_ID" "$SEEDS" > "$LOG" 2>&1 &
SWEEP_PID=$!
echo "$SWEEP_PID" > "$PIDFILE"

cat <<EOF

sweep started in the background
  pid:   $SWEEP_PID
  model: $MODEL_ID
  seeds: $SEEDS
  log:   $LOG

It keeps running if you close this terminal.
  monitor: tail -f $LOG
  status:  kill -0 $SWEEP_PID && echo running || echo finished
  stop:    kill $SWEEP_PID
EOF
