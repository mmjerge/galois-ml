#!/usr/bin/env bash
# Launch the experiment sweep in a detached tmux session so it survives
# terminal closes, editor restarts, and tool-call timeouts.
#
# Usage:
#   experiments/run_all.sh [MODEL_ID] [SEEDS]
#
# Then:
#   tmux attach -t galois        # watch live (Ctrl-b then d to detach)
#   tail -f experiments/outputs/run_all.log
#   tmux kill-session -t galois  # stop everything

set -euo pipefail

MODEL_ID="${1:-us.anthropic.claude-sonnet-4-5-20250929-v1:0}"
SEEDS="${2:-3}"
SESSION="galois"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG="$REPO/experiments/outputs/run_all.log"

mkdir -p "$REPO/experiments/outputs"

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "session '$SESSION' already running; attach with: tmux attach -t $SESSION"
  exit 1
fi

chmod +x "$REPO/experiments/_sweep.sh"

# Keep the pane alive after the sweep finishes so results stay visible.
tmux new-session -d -s "$SESSION" -c "$REPO" \
  "bash experiments/_sweep.sh '$MODEL_ID' '$SEEDS' 2>&1 | tee '$LOG'; echo; echo '(sweep finished; Ctrl-b then d to detach)'; exec bash"

echo "started tmux session '$SESSION'"
echo "  model: $MODEL_ID"
echo "  seeds: $SEEDS"
echo "  log:   $LOG"
echo
echo "watch: tmux attach -t $SESSION     (Ctrl-b then d to detach)"
echo "tail:  tail -f $LOG"
echo "stop:  tmux kill-session -t $SESSION"
