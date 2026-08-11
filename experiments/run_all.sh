#!/usr/bin/env bash
# Prepare a detached tmux session for the experiment sweep.
#
# Authentication is session-scoped (credentials are exported into the
# shell), so the sweep must run in a shell that was authenticated
# itself. This script therefore does NOT start the sweep: it creates a
# clean tmux session, and you authenticate inside it and launch the
# sweep there. The session survives terminal closes and tool timeouts.
#
# Usage:
#   experiments/run_all.sh [MODEL_ID] [SEEDS]
#
# Then:
#   tmux attach -t galois
#     <authenticate in that shell, e.g. your isengard/ada command>
#     ./experiments/_sweep.sh          # uses the defaults printed below
#   Ctrl-b then d                      # detach; sweep keeps running
#
#   tail -f experiments/outputs/run_all.log
#   tmux kill-session -t galois        # stop everything
#
# If you prefer a one-liner and your auth command is non-interactive,
# pass it via AUTH_CMD and the sweep will start automatically:
#   AUTH_CMD='your-auth-command' experiments/run_all.sh

set -euo pipefail

MODEL_ID="${1:-us.anthropic.claude-sonnet-4-5-20250929-v1:0}"
SEEDS="${2:-3}"
SESSION="galois"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG="$REPO/experiments/outputs/run_all.log"
AUTH_CMD="${AUTH_CMD:-}"

mkdir -p "$REPO/experiments/outputs"
chmod +x "$REPO/experiments/_sweep.sh"

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "session '$SESSION' already exists."
  echo "  attach: tmux attach -t $SESSION"
  echo "  reset:  tmux kill-server && $0 $*"
  exit 1
fi

# IMPORTANT: the tmux *server* caches its environment from whenever it
# first started. A server left over from an earlier, unauthenticated
# shell hands that stale environment to every new session, so
# authenticating inside the session can still fail. Killing the server
# (not merely the session) is the only reliable reset.
if tmux ls >/dev/null 2>&1; then
  echo "existing tmux server found; killing it so the new session gets a"
  echo "clean environment (stale cached credentials are the usual cause"
  echo "of 'security token invalid' after a successful login)."
  tmux kill-server 2>/dev/null || true
  sleep 1
fi

# Start a plain interactive shell in the repo. Nothing is scrubbed: the
# shell you authenticate in is the shell the sweep runs in.
tmux new-session -d -s "$SESSION" -c "$REPO"

SWEEP_CMD="./experiments/_sweep.sh '$MODEL_ID' '$SEEDS' 2>&1 | tee '$LOG'"

if [[ -n "$AUTH_CMD" ]]; then
  echo "running AUTH_CMD inside the session, then starting the sweep"
  tmux send-keys -t "$SESSION" "$AUTH_CMD" C-m
  tmux send-keys -t "$SESSION" "$SWEEP_CMD" C-m
  echo "started: auth + sweep are running in session '$SESSION'"
else
  # Pre-type the sweep command without running it, so after
  # authenticating you can start it with a single Enter.
  tmux send-keys -t "$SESSION" "clear; echo '>> authenticate in this shell, then press Enter to start the sweep'" C-m
  tmux send-keys -t "$SESSION" "$SWEEP_CMD"
  cat <<EOF
created tmux session '$SESSION' (sweep NOT started yet)
  model: $MODEL_ID
  seeds: $SEEDS
  log:   $LOG

Next:
  1) tmux attach -t $SESSION
  2) authenticate in that shell (your isengard/ada command)
  3) press Enter -- the sweep command is already typed at the prompt
  4) Ctrl-b then d to detach; it keeps running

Monitor:  tail -f $LOG
Stop:     tmux kill-session -t $SESSION
EOF
fi
