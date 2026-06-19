#!/bin/bash
# =============================================================================
# ZeroCausal Full-Dataset Evaluation Runner (SSH-Safe)
# =============================================================================
# Usage:
#   nohup bash run_all_evaluations.sh > logs/master_eval.log 2>&1 &
#   echo $! > logs/master_eval.pid
#
# Monitor:
#   tail -f logs/master_eval.log
#
# Check if running:
#   cat logs/master_eval.pid | xargs ps -p
# =============================================================================

VENV=".venv_zc/bin/python3"
LOG_DIR="logs"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

mkdir -p "$LOG_DIR"

echo "============================================================"
echo "  ZeroCausal Full-Dataset Evaluation"
echo "  Started: $(date)"
echo "  PID: $$"
echo "============================================================"
echo ""

# Save PID
echo $$ > "$LOG_DIR/master_eval.pid"

# Track failures
FAILURES=""

# -------------------------------------------------------------------
# Step 0: Preprocess BETH with per-host data
# -------------------------------------------------------------------
echo "[$(date)] Step 0: Preprocessing BETH dataset..."
$VENV 00c_preprocess_beth.py 2>&1 | tee "$LOG_DIR/beth_preprocess_${TIMESTAMP}.log"
if [ ${PIPESTATUS[0]} -ne 0 ]; then
    echo "FAILED: BETH preprocessing" | tee -a "$LOG_DIR/failures.log"
    FAILURES="$FAILURES beth_preprocess"
fi
echo ""

# -------------------------------------------------------------------
# Step 0b: Clear stale PCMCI caches (force fresh causal discovery)
# -------------------------------------------------------------------
echo "[$(date)] Clearing stale PCMCI caches..."
rm -f logs/pcmci_cache_streamspot_*.pkl logs/pcmci_cache_beth_*.pkl
echo "  Done."
echo ""

# -------------------------------------------------------------------
# Step 1: TC3 Evaluation
# -------------------------------------------------------------------
echo "[$(date)] Step 1/5: Evaluating TC3..."
$VENV 09_evaluate_additional_datasets.py \
    --dataset tc3 --baseline --simulate-drift \
    --std-floor 1.0 --run-name tc3_default \
    --checkpoint-interval 500 \
    2>&1 | tee "$LOG_DIR/tc3_eval_${TIMESTAMP}.log"
if [ ${PIPESTATUS[0]} -ne 0 ]; then
    echo "FAILED: TC3" | tee -a "$LOG_DIR/failures.log"
    FAILURES="$FAILURES tc3"
fi
echo ""

# -------------------------------------------------------------------
# Step 2: NODLINK Evaluation
# -------------------------------------------------------------------
echo "[$(date)] Step 2/5: Evaluating NODLINK..."
$VENV 09_evaluate_additional_datasets.py \
    --dataset nodlink --baseline \
    --std-floor 1.0 --run-name nodlink_default \
    --checkpoint-interval 500 \
    2>&1 | tee "$LOG_DIR/nodlink_eval_${TIMESTAMP}.log"
if [ ${PIPESTATUS[0]} -ne 0 ]; then
    echo "FAILED: NODLINK" | tee -a "$LOG_DIR/failures.log"
    FAILURES="$FAILURES nodlink"
fi
echo ""

# -------------------------------------------------------------------
# Step 3: StreamSpot Evaluation (FULL dataset, graph-aware split)
# -------------------------------------------------------------------
echo "[$(date)] Step 3/5: Evaluating STREAMSPOT (full 89K windows)..."
$VENV 09_evaluate_additional_datasets.py \
    --dataset streamspot --baseline \
    --train-limit 5000 \
    --run-name streamspot_default \
    --checkpoint-interval 500 \
    2>&1 | tee "$LOG_DIR/streamspot_eval_${TIMESTAMP}.log"
if [ ${PIPESTATUS[0]} -ne 0 ]; then
    echo "FAILED: STREAMSPOT" | tee -a "$LOG_DIR/failures.log"
    FAILURES="$FAILURES streamspot"
fi
echo ""

# -------------------------------------------------------------------
# Step 4: BETH Evaluation (per-host data)
# -------------------------------------------------------------------
echo "[$(date)] Step 4/5: Evaluating BETH (per-host data)..."
$VENV 09_evaluate_additional_datasets.py \
    --dataset beth --baseline \
    --run-name beth_default \
    --checkpoint-interval 500 \
    2>&1 | tee "$LOG_DIR/beth_eval_${TIMESTAMP}.log"
if [ ${PIPESTATUS[0]} -ne 0 ]; then
    echo "FAILED: BETH" | tee -a "$LOG_DIR/failures.log"
    FAILURES="$FAILURES beth"
fi
echo ""

# -------------------------------------------------------------------
# Step 5: Generate Plots & Compile Paper
# -------------------------------------------------------------------
echo "[$(date)] Step 5/5: Generating comparison plots..."
$VENV 10_plot_comparisons.py 2>&1 | tee "$LOG_DIR/plots_${TIMESTAMP}.log"

echo "[$(date)] Compiling LaTeX paper..."
pdflatex -interaction=nonstopmode ZeroCausal_Paper.tex > "$LOG_DIR/pdflatex_${TIMESTAMP}.log" 2>&1
pdflatex -interaction=nonstopmode ZeroCausal_Paper.tex > "$LOG_DIR/pdflatex2_${TIMESTAMP}.log" 2>&1
echo ""

# -------------------------------------------------------------------
# Final Summary
# -------------------------------------------------------------------
echo "============================================================"
echo "  EVALUATION COMPLETE"
echo "  Finished: $(date)"
echo "============================================================"
echo ""

# Print all AUC scores
echo "  Dataset Results:"
echo "  ─────────────────────────────────"
for ds in tc3 nodlink streamspot beth; do
    SUMMARY="$LOG_DIR/${ds}_default_summary.json"
    if [ -f "$SUMMARY" ]; then
        AUC=$($VENV -c "import json; d=json.load(open('$SUMMARY')); print(f\"{d['metrics']['auc']:.4f}\")" 2>/dev/null)
        echo "  $ds: AUC = $AUC"
    else
        echo "  $ds: NO SUMMARY FILE"
    fi
done
echo ""

if [ -n "$FAILURES" ]; then
    echo "  ⚠️  FAILURES: $FAILURES"
else
    echo "  ✅ All evaluations completed successfully!"
fi
echo ""

# Clean up PID file
rm -f "$LOG_DIR/master_eval.pid"
