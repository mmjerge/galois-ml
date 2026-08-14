#!/usr/bin/env bash
# The experiment sweep. Invoked inside tmux by run_all.sh, or directly in
# an authenticated shell.
#
# Args: MODEL_ID SEEDS
#
# Ordering matters: the most informative experiments run FIRST, because a
# long overnight sweep may outlive the credentials it started with. Each
# stage writes results incrementally, so partial progress is never lost.
#
# If credentials expire mid-sweep the run does not burn through the
# remaining stages failing. It pauses and re-probes Bedrock every few
# minutes, so re-authenticating (in any shell, since the credentials file
# is shared) lets the sweep resume where it left off.

set -uo pipefail

MODEL_ID="${1:?model id required}"
SEEDS="${2:-3}"

# Pick an interpreter that can actually import the package and the agent
# dependencies. The local macOS environment keeps them in anaconda while
# EC2 hosts use the system python3, so the path must not be hardcoded.
# Override with GALOIS_PYTHON=... if needed.
select_python() {
  local candidates=("${GALOIS_PYTHON:-}" /opt/anaconda3/bin/python3 python3 python)
  for candidate in "${candidates[@]}"; do
    [[ -z "$candidate" ]] && continue
    command -v "$candidate" >/dev/null 2>&1 || continue
    if "$candidate" -c "import galois_ml, strands, boto3" >/dev/null 2>&1; then
      echo "$candidate"; return 0
    fi
  done
  # fall back to anything that imports the package, so we can still fail
  # with a useful dependency message rather than a confusing one
  for candidate in "${candidates[@]}"; do
    [[ -z "$candidate" ]] && continue
    command -v "$candidate" >/dev/null 2>&1 || continue
    if "$candidate" -c "import galois_ml" >/dev/null 2>&1; then
      echo "$candidate"; return 0
    fi
  done
  echo "python3"
}

PYTHON="$(select_python)"
echo "using interpreter: $PYTHON"

# Python block-buffers stdout when it is a file rather than a terminal,
# which makes a redirected log look frozen. Force line buffering.
export PYTHONUNBUFFERED=1

# Unset ONLY the Bedrock bearer token: boto3 prefers it over ordinary
# credentials, so a stale one (e.g. hardcoded in ~/.zshrc) causes
# "Bearer Token has expired" regardless of how recently you logged in.
if [[ -n "${AWS_BEARER_TOKEN_BEDROCK:-}" ]]; then
  echo "note: unsetting AWS_BEARER_TOKEN_BEDROCK for this run"
  unset AWS_BEARER_TOKEN_BEDROCK
fi

bedrock_ok() {
  "$PYTHON" -c "
from galois_ml.llm_eval import BedrockModel
BedrockModel('$MODEL_ID', max_tokens=16)('ok')
" >/dev/null 2>&1
}

# Block until Bedrock answers, re-probing periodically. Returns non-zero
# only if it never recovers within the timeout.
wait_for_credentials() {
  local waited=0 interval=300 max_wait=43200  # 5 min probe, 12 h ceiling
  while ! bedrock_ok; do
    if (( waited >= max_wait )); then
      echo "!! credentials never returned after $((max_wait / 3600))h; stopping"
      return 1
    fi
    echo "!! Bedrock unavailable (likely expired session). Re-authenticate;"
    echo "   re-probing every $((interval / 60)) min. Waited $((waited / 60)) min so far."
    sleep "$interval"
    waited=$((waited + interval))
  done
  echo "== credentials OK, continuing =="
  return 0
}

stage() {
  local name="$1"; shift
  echo
  echo "=============================================================="
  echo "== $name  ($(date '+%Y-%m-%d %H:%M:%S'))"
  echo "=============================================================="
  if ! bedrock_ok; then
    wait_for_credentials || return 1
  fi
  "$@"
  echo "-- $name finished ($(date '+%H:%M:%S')) --"
}

echo "=== preflight: verifying Bedrock access ==="
if ! bedrock_ok; then
  echo "PREFLIGHT FAILED: authenticate in this shell first."
  exit 1
fi
echo "bedrock ok"

# 1. Tool poisoning, ALL channels, theory withheld so the tool channel is
#    load-bearing. This is the corrected harm measurement and the most
#    informative single experiment, so it runs first.
stage "1/5 tool poisoning: all channels, theory withheld (attack harm)" \
  "$PYTHON" experiments/tool_poisoning_eval.py --model-id "$MODEL_ID" \
    --tasks quartic --attacks none data verification injection combined impossible \
    --seeds "$SEEDS" --temperature 0.7 --no-theory-in-brief \
    --out-suffix _harm

# 2. The same channels with the theory present, which measures robustness
#    through channel redundancy rather than harm.
stage "2/5 tool poisoning: all channels, theory present (redundancy)" \
  "$PYTHON" experiments/tool_poisoning_eval.py --model-id "$MODEL_ID" \
    --tasks quartic --attacks none data verification injection combined impossible \
    --seeds "$SEEDS" --temperature 0.7 \
    --out-suffix _redundancy

# 3. Correlated failure: does redundancy buy independence?
stage "3/5 conformity: 5 independent agents per task" \
  "$PYTHON" experiments/conformity_eval.py --model-id "$MODEL_ID" \
    --tasks quartic sextic --agents 5 --temperature 1.0

# 4. Agent-level compromise and verification value.
stage "4/5 multi-agent conditions (verification value, sabotage)" \
  "$PYTHON" experiments/multi_agent_eval.py --model-id "$MODEL_ID" \
    --tasks quartic sextic degree21

# 5. Figures and aggregates (no Bedrock needed).
echo
echo "== 5/5 figures and summary =="
"$PYTHON" experiments/plot_results.py

echo
echo "=== SWEEP COMPLETE ($(date '+%Y-%m-%d %H:%M:%S')) ==="
