#!/usr/bin/env bash
# The experiment sweep itself. Invoked inside tmux by run_all.sh.
# Args: MODEL_ID SEEDS
set -uo pipefail

MODEL_ID="${1:?model id required}"
SEEDS="${2:-3}"
PYTHON="/opt/anaconda3/bin/python3"

# Python block-buffers stdout when it is a file rather than a terminal,
# which makes a redirected log look frozen for many minutes even while
# the sweep is progressing normally. Force line buffering so `tail -f`
# reflects real-time progress.
export PYTHONUNBUFFERED=1

# Unset ONLY the Bedrock bearer token. boto3 gives it precedence over
# ordinary AWS credentials for Bedrock calls, so a stale one (e.g. a
# hardcoded `export AWS_BEARER_TOKEN_BEDROCK=...` in ~/.zshrc) causes
# "Bearer Token has expired" no matter how recently you authenticated.
# The STS credentials (AWS_ACCESS_KEY_ID / SECRET / SESSION_TOKEN) are
# deliberately preserved: session-scoped auth provides them and they are
# what we actually want to use.
if [[ -n "${AWS_BEARER_TOKEN_BEDROCK:-}" ]]; then
  echo "note: unsetting AWS_BEARER_TOKEN_BEDROCK for this run"
  echo "      (it overrides your session credentials; see ~/.zshrc)"
  unset AWS_BEARER_TOKEN_BEDROCK
fi

echo "=== preflight: verifying Bedrock access ==="
if ! "$PYTHON" -c "
from galois_ml.llm_eval import BedrockModel
m = BedrockModel('$MODEL_ID', max_tokens=64)
print('bedrock ok:', m('Reply with the single word: ready').strip()[:40])
"; then
  echo
  echo "PREFLIGHT FAILED: no valid Bedrock credentials in THIS shell."
  echo "Authenticate here (you are already inside the tmux session), then"
  echo "rerun:  ./experiments/_sweep.sh '$MODEL_ID' '$SEEDS'"
  exit 1
fi

echo
echo "=== 1/4 tool poisoning, theory withheld (attack harm) ==="
"$PYTHON" experiments/tool_poisoning_eval.py --model-id "$MODEL_ID" \
  --tasks quartic --seeds "$SEEDS" --temperature 0.7 \
  --no-theory-in-brief --out-suffix _harm

echo
echo "=== 2/4 tool poisoning, theory present (redundancy) ==="
"$PYTHON" experiments/tool_poisoning_eval.py --model-id "$MODEL_ID" \
  --tasks quartic --seeds "$SEEDS" --temperature 0.7 \
  --out-suffix _redundancy

echo
echo "=== 3/4 multi-agent conditions (verification value, sabotage) ==="
"$PYTHON" experiments/multi_agent_eval.py --model-id "$MODEL_ID" \
  --tasks quartic sextic degree21

echo
echo "=== 4/4 figures and summary ==="
"$PYTHON" experiments/plot_results.py

echo
echo "=== SWEEP COMPLETE ==="
