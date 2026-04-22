#!/usr/bin/env bash
# Evaluate llama-2-70b-chat-hf (Vertex MaaS) on all three datasets.
# Run this inside a tmux window; it chains three jobs sequentially.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SCRIPTS="$ROOT/scripts"
DATA="$ROOT/data"

MODEL="ignored"
BACKEND="vertex_endpoint"
PROJECT="cs-hu-lab-research-63f8"
LOCATION="us-west1"
ENDPOINT_ID="1463056351407112192"
GRADER="gemini-3.1-pro-preview"
WORKERS=4

run_eval() {
  local dataset_name="$1"
  local in_file="$2"
  local out_file="$3"
  echo ""
  echo "========================================"
  echo " llama2 | $dataset_name"
  echo " input : $in_file"
  echo " output: $out_file"
  echo "========================================"
  python3 "$SCRIPTS/automated_eval.py" \
    --input  "$in_file" \
    --output "$out_file" \
    --target-model       "$MODEL" \
    --target-backend     "$BACKEND" \
    --vertex-project     "$PROJECT" \
    --vertex-location    "$LOCATION" \
    --vertex-endpoint-id "$ENDPOINT_ID" \
    --grader-model       "$GRADER" \
    --grader-max-tokens  4096 \
    --max-workers        "$WORKERS" \
    --continue-on-error
}

run_eval "isolated_kept" \
  "$DATA/stage3_1_isolated_kept.jsonl" \
  "$DATA/eval_llama2_isolated_kept.jsonl"

run_eval "isolated_nc6" \
  "$DATA/stage3_1_isolated_no_check6.jsonl" \
  "$DATA/eval_llama2_isolated_nc6.jsonl"

run_eval "audit_t4096" \
  "$DATA/stage3_1_audit_t4096_new.jsonl" \
  "$DATA/eval_llama2_audit_t4096.jsonl"

echo ""
echo "=== llama2: ALL DONE ==="
