#!/usr/bin/env bash
# The experiment sweep itself. Invoked inside tmux by run_all.sh.
# Args: MODEL_ID SEEDS
set -uo pipefail

MODEL_ID="${1:?model id required}"
SEEDS="${2:-3}"
PYTHON="/opt/anaconda3/bin/python3"

# Scrub stale Bedrock/STS env credentials so boto3 uses ~/.aws instead.
unset AWS_BEARER_TOKEN_BEDROCK AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY AWS_SESSION_TOKEN

echo "=== preflight: verifying Bedrock access ==="
if ! "$PYTHON" -c "
from galois_ml.llm_eval import BedrockModel
m = BedrockModel('$MODEL_ID', max_tokens=64)
print('bedrock ok:', m('Reply with the single word: ready').strip()[:40])
"; then
  echo "PREFLIGHT FAILED: re-authenticate (isengard), then rerun experiments/run_all.sh"
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
