#!/usr/bin/env bash
# Manage the sweep running on the dedicated EC2 instance.
#
# The instance uses an IAM instance profile with AmazonBedrockFullAccess,
# so Bedrock credentials never expire there. That is the reason to run
# long sweeps on it rather than on a laptop with a short-lived session.
#
# Usage:
#   experiments/remote.sh status    progress from the result files
#   experiments/remote.sh log       tail the sweep log
#   experiments/remote.sh watch     follow the log live
#   experiments/remote.sh attach    attach to the remote tmux session
#   experiments/remote.sh pull      copy results and figures back here
#   experiments/remote.sh sync      push local code changes to the host
#   experiments/remote.sh restart   resync code and restart the sweep
#   experiments/remote.sh stop      kill the sweep (instance keeps running)
#   experiments/remote.sh ssh       open a shell on the host

set -uo pipefail

HOST="${GALOIS_HOST:-98.83.148.117}"
KEY="${GALOIS_KEY:-$HOME/.ssh/jerge-ec2.pem}"
INSTANCE_ID="${GALOIS_INSTANCE_ID:-i-063cef98877d75def}"
REMOTE_DIR="galois-ml"
MODEL="${GALOIS_MODEL:-us.anthropic.claude-sonnet-4-5-20250929-v1:0}"
SEEDS="${GALOIS_SEEDS:-3}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

sshx() { ssh -o StrictHostKeyChecking=accept-new -i "$KEY" "ubuntu@$HOST" "$@"; }

case "${1:-status}" in
  status)
    sshx "cd $REMOTE_DIR && bash experiments/watch_sweep.sh --once"
    ;;
  log)   sshx "tail -40 $REMOTE_DIR/experiments/outputs/run_all.log" ;;
  watch) sshx "tail -f $REMOTE_DIR/experiments/outputs/run_all.log" ;;
  attach)
    echo "attaching to remote tmux (Ctrl-b then d to detach)"
    ssh -t -i "$KEY" "ubuntu@$HOST" "tmux attach -t galois"
    ;;
  pull)
    mkdir -p "$REPO/experiments/outputs" "$REPO/experiments/figures"
    rsync -az -e "ssh -i $KEY" \
      "ubuntu@$HOST:$REMOTE_DIR/experiments/outputs/" \
      "$REPO/experiments/outputs/"
    rsync -az -e "ssh -i $KEY" \
      "ubuntu@$HOST:$REMOTE_DIR/experiments/figures/" \
      "$REPO/experiments/figures/" 2>/dev/null || true
    echo "pulled results into experiments/outputs and experiments/figures"
    ;;
  sync)
    rsync -az --exclude '.git' --exclude 'experiments/outputs' \
      --exclude 'benchmarks' --exclude '__pycache__' --exclude '*.egg-info' \
      --exclude '.lmfdb_cache' --exclude 'experiments/figures' \
      -e "ssh -i $KEY" "$REPO/" "ubuntu@$HOST:$REMOTE_DIR/"
    echo "code synced to $HOST"
    ;;
  restart)
    "$0" sync
    sshx "cd $REMOTE_DIR && tmux kill-session -t galois 2>/dev/null; \
      tmux new-session -d -s galois -c ~/$REMOTE_DIR \
      'bash experiments/_sweep.sh $MODEL $SEEDS 2>&1 | tee experiments/outputs/run_all.log; echo SWEEP_EXITED; exec bash'"
    echo "sweep restarted on $HOST"
    ;;
  stop)
    sshx "tmux kill-session -t galois 2>/dev/null; echo 'sweep stopped'"
    ;;
  ssh)  ssh -t -i "$KEY" "ubuntu@$HOST" ;;
  info)
    echo "host:     $HOST"
    echo "instance: $INSTANCE_ID (g5.xlarge, A10G, us-east-1c)"
    echo "role:     canopy-bedrock-ec2 (AmazonBedrockFullAccess)"
    echo "remote:   ~/$REMOTE_DIR"
    ;;
  *)
    sed -n '2,30p' "$0"; exit 1
    ;;
esac
