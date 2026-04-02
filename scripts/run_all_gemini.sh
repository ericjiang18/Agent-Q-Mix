#!/bin/bash
# ============================================================
#  Run COMPLETE pipeline with Gemini 3.1 Flash Lite Preview:
#    Train (unified, 3 datasets interleaved) -> Test (7 benchmarks)
#
#  Usage:
#    bash scripts/run_all_gemini.sh
#    DATA_LIMIT=5 bash scripts/run_all_gemini.sh   # quick test
# ============================================================
set -e
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD:$PYTHONPATH"

# Ensure we use the conda with torch installed
source "$HOME/miniconda3/etc/profile.d/conda.sh" 2>/dev/null && conda activate base 2>/dev/null || true

export BASE_URL="${BASE_URL:-https://generativelanguage.googleapis.com/v1beta/openai}"
export API_KEY="${GEMINI_API_KEY:-PUT_YOURS_HERE}"
export LLM_MODEL="${LLM_MODEL:-gemini-3.1-flash-lite-preview}"

LLM="$LLM_MODEL"
EPISODES=${TRAIN_EPISODES:-50}
DEVICE=${DEVICE:-"cpu"}
LIMIT=${DATA_LIMIT:-""}
RESULT_DIR="result_gemini"
CKPT_DIR="checkpoints_gemini"

mkdir -p "$RESULT_DIR" "$CKPT_DIR"

echo "========================================="
echo "  QMIX: FULL PIPELINE (Gemini Flash Lite)"
echo "  Model     : $LLM"
echo "  Base URL  : $BASE_URL"
echo "  Results   : $RESULT_DIR"
echo "  Checkpoint: $CKPT_DIR/qmix_unified.pt"
echo "========================================="

# ── PHASE 1: UNIFIED TRAINING ───────────────────────────────
echo ""
echo ">>> PHASE 1: UNIFIED TRAINING (3 datasets interleaved)"
echo ""

python3 -m experiments.run_qmix_train \
    --dataset "livecodebench_testgen,mmlu_pro,aime_2024" \
    --split test \
    --llm_name "$LLM" \
    --num_episodes "$EPISODES" \
    --num_rounds 2 \
    --batch_size 8 \
    --data_limit 15 \
    --token_penalty 0.075 \
    --accuracy_weight 1.50 \
    --device "$DEVICE" \
    --save_path "$CKPT_DIR/qmix_unified.pt" \
    --log_interval 10 \
    || echo "  [WARN] Training exited with non-zero code (checkpoint may still be saved)"

if [ ! -f "$CKPT_DIR/qmix_unified.pt" ]; then
    echo "  [ERROR] Checkpoint not found: $CKPT_DIR/qmix_unified.pt"
    exit 1
fi

echo ""
echo "  TRAINING COMPLETE -- single checkpoint: $CKPT_DIR/qmix_unified.pt"
echo ""

# ── PHASE 2: TESTING ────────────────────────────────────────
echo ">>> PHASE 2: TESTING (8 benchmarks)"
echo ""

CKPT="$CKPT_DIR/qmix_unified.pt"
LIMIT_ARG=""
[ -n "$LIMIT" ] && LIMIT_ARG="--limit $LIMIT"

# Coding
#echo "[1/7] LiveCodeBench..."
#python3 -m experiments.run_qmix_eval \
#    --dataset livecodebench --llm_name "$LLM" --num_rounds 2 \
#    --model_path "$CKPT" \
#    --mode qmix --output_path "$RESULT_DIR/livecodebench_qmix.json" \
#    $LIMIT_ARG

echo "[2/7] HumanEval..."
python3 -m experiments.run_qmix_eval \
    --dataset humaneval --llm_name "$LLM" --num_rounds 2 \
    --model_path "$CKPT" \
    --mode qmix --output_path "$RESULT_DIR/humaneval_qmix.json" \
    $LIMIT_ARG

# Agentic
echo "[3/7] MMLU-Pro..."
python3 -m experiments.run_qmix_eval \
    --dataset mmlu_pro --split validation --llm_name "$LLM" --num_rounds 2 \
    --model_path "$CKPT" \
    --mode qmix --output_path "$RESULT_DIR/mmlu_pro_qmix.json" \
    $LIMIT_ARG

# Math
echo "[4/7] AIME 2025..."
python3 -m experiments.run_qmix_eval \
    --dataset aime_2025 --split train --llm_name "$LLM" --num_rounds 3 \
    --model_path "$CKPT" \
    --mode qmix --output_path "$RESULT_DIR/aime2025_qmix.json"

echo "[5/7] AIME 2026..."
python3 -m experiments.run_qmix_eval \
    --dataset aime_2026 --split train --llm_name "$LLM" --num_rounds 3 \
    --model_path "$CKPT" \
    --mode qmix --output_path "$RESULT_DIR/aime2026_qmix.json"

echo "[6/7] Beyond-AIME..."
python3 -m experiments.run_qmix_eval \
    --dataset beyond_aime --split test --llm_name "$LLM" --num_rounds 3 \
    --model_path "$CKPT" \
    --mode qmix --output_path "$RESULT_DIR/beyond_aime_qmix.json"

echo "[7/8] HMMT Feb 2025..."
python3 -m experiments.run_qmix_eval \
    --dataset hmmt_2025 --split train --llm_name "$LLM" --num_rounds 3 \
    --model_path "$CKPT" \
    --mode qmix --output_path "$RESULT_DIR/hmmt2025_qmix.json"

# HLE
echo "[8/8] HLE (MCQ, first 250)..."
python3 -m experiments.run_qmix_eval \
    --dataset hle --split test --llm_name "$LLM" --num_rounds 2 \
    --limit 250 \
    --model_path "$CKPT" \
    --mode qmix --output_path "$RESULT_DIR/hle_qmix.json"

echo ""
echo "========================================="
echo "  ALL BENCHMARKS COMPLETE (Gemini Flash Lite)"
echo "  Generating summary …"
echo "========================================="
echo ""
python3 summarize_results.py "$RESULT_DIR"
