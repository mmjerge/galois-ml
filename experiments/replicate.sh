#!/usr/bin/env bash
# Full-scale replication at the statistical power the measurement needs.
#
# Observed throughput is ~22s per scored decision, so 400 decisions per
# cell across a 16-cell grid is ~40h serial. Cells are independent, so
# this shards them across parallel workers: one tmux session per shard,
# each writing to its own results file. Merge with report_sequential.py.
#
# Power: 400 decisions per cell gives a binomial SE of ~0.015 at p=0.9,
# i.e. +/-0.03 at 95%. The earlier 30-decision cells were +/-0.15, which
# could not resolve a 0.567 vs 0.900 difference, let alone anything finer.
#
# Usage: experiments/replicate.sh [WORKERS]
#   experiments/replicate.sh 8
#
# Monitor:  tmux ls; tail -f experiments/outputs/rep_*.log
# Merge:    python experiments/report_sequential.py

set -uo pipefail

WORKERS="${1:-8}"
cd "$(dirname "${BASH_SOURCE[0]}")/.."
OUT="experiments/outputs"
mkdir -p "$OUT"

MODELS=("us.anthropic.claude-opus-4-8" "us.anthropic.claude-sonnet-4-6")
ROLES=("listener" "credulous")
RATES=(0.0 0.10 0.25 0.50)
# 40 episodes x 10 decisions = 400 decisions per cell
SEEDS=40
DECISIONS=10

PY="python3"
command -v /opt/anaconda3/bin/python3 >/dev/null 2>&1 && \
  /opt/anaconda3/bin/python3 -c "import strands" 2>/dev/null && \
  PY=/opt/anaconda3/bin/python3

# Build the shard list: one shard per (model, role), each covering all
# rates, so a shard is a self-contained curve and partial results are
# always interpretable rather than a scatter of unrelated cells.
shards=()
for m in "${MODELS[@]}"; do
  for r in "${ROLES[@]}"; do
    shards+=("$m|$r")
  done
done

echo "replication plan"
echo "  cells:     ${#MODELS[@]} models x ${#ROLES[@]} roles x ${#RATES[@]} rates"
echo "  per cell:  $SEEDS episodes x $DECISIONS decisions = $((SEEDS*DECISIONS))"
echo "  shards:    ${#shards[@]} (one per model/role curve)"
echo "  workers:   $WORKERS"
echo

launched=0
for shard in "${shards[@]}"; do
  model="${shard%%|*}"
  role="${shard##*|}"
  tag="$(echo "${model##*claude-}_${role}" | tr -c 'a-zA-Z0-9_' '_')"
  session="rep_${tag}"
  outfile="$OUT/rep_${tag}.json"

  # throttle to WORKERS concurrent sessions
  while [[ "$(tmux ls 2>/dev/null | grep -c '^rep_')" -ge "$WORKERS" ]]; do
    sleep 30
  done

  tmux kill-session -t "$session" 2>/dev/null
  tmux new-session -d -s "$session" -c "$PWD" \
    "export PYTHONUNBUFFERED=1; unset AWS_BEARER_TOKEN_BEDROCK; \
     $PY experiments/sequential_gullibility_eval.py \
       --task quartic --models '$model' --roles '$role' \
       --rates ${RATES[*]} --seeds $SEEDS --decisions $DECISIONS \
       --calls-per-decision 30 --out '$outfile' \
       2>&1 | tee '$OUT/rep_${tag}.log'; \
     echo REP_DONE_${tag}; exec bash"
  launched=$((launched+1))
  echo "launched $session -> $outfile"
  sleep 2
done

echo
echo "$launched shards running; ~$((SEEDS*DECISIONS*${#RATES[@]}*22/3600))h each"
echo "monitor: tmux ls | grep rep_"
echo "merge:   $PY experiments/report_sequential.py"
