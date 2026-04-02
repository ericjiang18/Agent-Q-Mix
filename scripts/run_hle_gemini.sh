#!/bin/bash
# ============================================================
#  Humanity's Last Exam (MCQ only, first 250) — Gemini Flash Lite
#  Uses the unified QMIX checkpoint from run_all_gemini.sh
#
#  Usage:
#    bash scripts/run_hle_gemini.sh
# ============================================================
set -e
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD:$PYTHONPATH"

source "$HOME/miniconda3/etc/profile.d/conda.sh" 2>/dev/null && conda activate base 2>/dev/null || true

export BASE_URL="${BASE_URL:-https://generativelanguage.googleapis.com/v1beta/openai}"
export API_KEY="${GEMINI_API_KEY:-PUT_YOURS_HERE}"
export LLM_MODEL="${LLM_MODEL:-gemini-3.1-flash-lite-preview}"

LLM="$LLM_MODEL"
CKPT="checkpoints_gemini/qmix_unified.pt"
RESULT_DIR="result_hle_gemini"

mkdir -p "$RESULT_DIR"

echo "========================================="
echo "  HLE Benchmark (MCQ, 250) — QMIX"
echo "  Model     : $LLM"
echo "  Checkpoint: $CKPT"
echo "  Results   : $RESULT_DIR"
echo "========================================="

if [ ! -f "$CKPT" ]; then
    echo "[ERROR] Checkpoint not found: $CKPT"
    echo "Run scripts/run_all_gemini.sh first to train."
    exit 1
fi

echo ""
echo "[QMIX] HLE (cais/hle, MCQ only, first 250)..."
python3 -m experiments.run_qmix_eval \
    --dataset hle \
    --split test \
    --llm_name "$LLM" \
    --num_rounds 3 \
    --limit 250 \
    --model_path "$CKPT" \
    --mode qmix \
    --output_path "$RESULT_DIR/hle_qmix.json"

echo ""
echo "========================================="
echo "  HLE COMPLETE"
echo "========================================="
echo ""
python3 summarize_results.py "$RESULT_DIR"
