#!/usr/bin/env bash
# Sequential experiment pipeline. One session, one stage at a time.
#
# Replaces an earlier design that ran each stage in its own tmux session
# waiting on a marker grepped from the previous stage's log. That was
# broken two ways: the marker was echoed after the `| tee`, so it never
# reached the log, and `exec bash` kept finished sessions alive, so the
# "is the predecessor still running?" test never went false. Every stage
# waited forever.
#
# Here each stage writes a sentinel file on success and is skipped if that
# sentinel already exists, so the pipeline is resumable and idempotent:
# rerun it after a failure and it picks up where it stopped.
#
# Usage: experiments/_pipeline.sh [MODEL_ID]
#   force a stage to rerun by deleting its sentinel in outputs/.done/

set -uo pipefail

MODEL="${1:-us.anthropic.claude-sonnet-4-5-20250929-v1:0}"
cd "$(dirname "${BASH_SOURCE[0]}")/.."
OUTPUTS="experiments/outputs"
DONE="$OUTPUTS/.done"
mkdir -p "$DONE"

export PYTHONUNBUFFERED=1
[[ -n "${AWS_BEARER_TOKEN_BEDROCK:-}" ]] && unset AWS_BEARER_TOKEN_BEDROCK

PY="python3"
command -v /opt/anaconda3/bin/python3 >/dev/null 2>&1 && \
  /opt/anaconda3/bin/python3 -c "import strands" 2>/dev/null && \
  PY=/opt/anaconda3/bin/python3

run_stage() {
  local name="$1"; shift
  if [[ -f "$DONE/$name" ]]; then
    echo "== SKIP $name (already complete) =="
    return 0
  fi
  echo
  echo "=============================================================="
  echo "== $name  ($(date '+%Y-%m-%d %H:%M:%S'))"
  echo "=============================================================="
  if "$@" 2>&1 | tee "$OUTPUTS/${name}.log"; then
    # PIPESTATUS[0] is the command's status, not tee's
    if [[ "${PIPESTATUS[0]}" -eq 0 ]]; then
      touch "$DONE/$name"
      echo "-- $name COMPLETE ($(date '+%H:%M:%S')) --"
      return 0
    fi
  fi
  echo "!! $name FAILED (no sentinel written; rerun to retry) !!"
  return 1
}

echo "pipeline start $(date '+%Y-%m-%d %H:%M:%S'), interpreter $PY"

run_stage multi_agent_seeded "$PY" experiments/multi_agent_eval.py \
  --model-id "$MODEL" --tasks sextic degree21 --seeds 5 --temperature 0.7

run_stage poison_harm_v2 "$PY" experiments/tool_poisoning_eval.py \
  --model-id "$MODEL" --tasks sextic degree21 --seeds 5 --temperature 0.7 \
  --no-theory-in-brief --out-suffix _harm_v2

run_stage poison_redundancy_v2 "$PY" experiments/tool_poisoning_eval.py \
  --model-id "$MODEL" --tasks sextic degree21 --seeds 5 --temperature 0.7 \
  --out-suffix _redundancy_v2

run_stage cross_model "$PY" experiments/cross_model_eval.py \
  --tasks sextic --attacks none impossible combined --seeds 3

run_stage gullibility "$PY" experiments/gullibility_eval.py \
  --task sextic --attack data --rates 0.0 0.10 0.25 0.50 --seeds 3

echo
echo "== figures =="
"$PY" experiments/plot_gullibility.py || true
"$PY" experiments/plot_corroboration.py || true
"$PY" experiments/plot_results.py || true

echo
echo "=== PIPELINE COMPLETE ($(date '+%Y-%m-%d %H:%M:%S')) ==="
echo "stages complete: $(ls "$DONE" | tr '\n' ' ')"
