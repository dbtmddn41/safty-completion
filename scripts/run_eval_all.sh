#!/usr/bin/env bash
# Launch all 9 eval jobs (3 models × 3 datasets) in separate tmux sessions.
#
# Usage:
#   bash scripts/run_eval_all.sh
#
# Creates three tmux sessions, one per model:
#   safeval-llama2        — llama2 via Vertex endpoint
#   safeval-gemini-flash  — gemini-3.1-flash-lite-preview via Gemini API
#   safeval-gemini-pro    — gemini-3.1-pro-preview via Gemini API
#
# Within each session, three datasets are evaluated sequentially:
#   1. stage3_1_isolated_kept.jsonl
#   2. stage3_1_isolated_no_check6.jsonl  (check6-relaxed)
#   3. stage3_1_audit_t4096_new.jsonl
#
# Attach with:
#   tmux attach -t safeval-llama2
#   tmux attach -t safeval-gemini-flash
#   tmux attach -t safeval-gemini-pro
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SCRIPTS="$ROOT/scripts"
DATA="$ROOT/data"

# ── Step 1: create check6-relaxed dataset if absent ───────────────────────
NC6="$DATA/stage3_1_isolated_no_check6.jsonl"
if [[ ! -f "$NC6" ]]; then
  echo "→ Creating check6-relaxed dataset..."
  python3 "$SCRIPTS/make_isolated_nc6.py"
  echo "  Done."
else
  echo "→ check6-relaxed dataset already exists ($NC6)."
fi

# ── Step 2: make scripts executable ───────────────────────────────────────
chmod +x "$SCRIPTS/run_llama2_eval.sh" \
         "$SCRIPTS/run_gemini_flash_eval.sh" \
         "$SCRIPTS/run_gemini_pro_eval.sh"

# ── Step 3: kill old sessions if any ──────────────────────────────────────
for sess in safeval safeval-llama2 safeval-gemini-flash safeval-gemini-pro; do
  tmux kill-session -t "$sess" 2>/dev/null || true
done

# ── Step 4: start one session per model ───────────────────────────────────
tmux new-session -d -s "safeval-llama2" \
  "bash '$SCRIPTS/run_llama2_eval.sh' 2>&1 | tee '$DATA/eval_llama2.log'"

tmux new-session -d -s "safeval-gemini-flash" \
  "bash '$SCRIPTS/run_gemini_flash_eval.sh' 2>&1 | tee '$DATA/eval_gemini_flash_lite.log'"

tmux new-session -d -s "safeval-gemini-pro" \
  "bash '$SCRIPTS/run_gemini_pro_eval.sh' 2>&1 | tee '$DATA/eval_gemini_pro.log'"

echo ""
echo "3 tmux sessions started (--max-workers 4 each → 4 rows in parallel per model)."
echo "  tmux attach -t safeval-llama2"
echo "  tmux attach -t safeval-gemini-flash"
echo "  tmux attach -t safeval-gemini-pro"
echo ""
echo "Logs:"
echo "  $DATA/eval_llama2.log"
echo "  $DATA/eval_gemini_flash_lite.log"
echo "  $DATA/eval_gemini_pro.log"
