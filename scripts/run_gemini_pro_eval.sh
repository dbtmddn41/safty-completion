#!/usr/bin/env bash
# Evaluate gemini-3.1-pro-preview on all three datasets.
# Run this inside a tmux window; it chains three jobs sequentially.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SCRIPTS="$ROOT/scripts"
DATA="$ROOT/data"

MODEL="gemini-3.1-pro-preview"
BACKEND="gemini"
GRADER="gemini-3.1-pro-preview"
WORKERS=4

run_eval() {
  local dataset_name="$1"
  local in_file="$2"
  local out_file="$3"
  echo ""
  echo "========================================"
  echo " gemini-pro | $dataset_name"
  echo " input : $in_file"
  echo " output: $out_file"
  echo "========================================"
  python3 "$SCRIPTS/automated_eval.py" \
    --input  "$in_file" \
    --output "$out_file" \
    --target-model   "$MODEL" \
    --target-backend "$BACKEND" \
    --grader-model   "$GRADER" \
    --max-workers    "$WORKERS"
}

run_eval "isolated_kept" \
  "$DATA/stage3_1_isolated_kept.jsonl" \
  "$DATA/eval_gemini_pro_isolated_kept.jsonl"

run_eval "isolated_nc6" \
  "$DATA/stage3_1_isolated_no_check6.jsonl" \
  "$DATA/eval_gemini_pro_isolated_nc6.jsonl"

run_eval "audit_t4096" \
  "$DATA/stage3_1_audit_t4096_new.jsonl" \
  "$DATA/eval_gemini_pro_audit_t4096.jsonl"

echo ""
echo "=== gemini-pro: ALL DONE ==="
