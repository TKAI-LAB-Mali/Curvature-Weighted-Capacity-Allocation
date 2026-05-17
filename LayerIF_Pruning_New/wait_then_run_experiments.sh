#!/bin/bash

CURRENT_SCRIPT="run_gemma_alpha_baseline.sh"
NEXT_SCRIPT="/data/mdl-layerIF/LayerIF_Pruning_New/run_gemma_uniform_baseline.sh"

# Directory where run_mistral_mdl.sh saves results, and total number of jobs
# Total = 4 families × 5 datasets × 3 methods = 60
RESULTS_DIR="results/alpha_baseline/"
TOTAL_JOBS=3

# ---------------------------------------------------------------------------
# Completion signal — toggle between two options:
#
#   "norms"      — looks for "[norms] Saved after-pruning norms" in log.txt
#                  Best when DRY_RUN=false and --track_norms is passed.
#                  This is the last line main_mistral.py writes before exit,
#                  so it reliably means the full job finished.
#
#   "zero_shot"  — looks for zero_shot_*.json files written by main_mistral.py
#                  after eval_zero_shot completes. Best when --eval_zero_shot
#                  is passed (DRY_RUN=false full runs).
#                  Not present in dry runs.
#
# ---------------------------------------------------------------------------
COMPLETION_SIGNAL="zero_shot"  # set to "norms" or "zero_shot"

echo "=================================================="
echo "Searching for running job ($CURRENT_SCRIPT)..."
echo "Using completion signal: $COMPLETION_SIGNAL"
echo "=================================================="

CURRENT_PID=$(pgrep -f "$CURRENT_SCRIPT" | head -1)

if [ -z "$CURRENT_PID" ]; then
  echo "WARNING: No running job found. Launching next script immediately..."
else
  echo "Found job with PID: $CURRENT_PID. Waiting for it to finish..."

  while kill -0 "$CURRENT_PID" 2>/dev/null; do

    # Count completed jobs using the selected signal.
    # Directory layout:
    #   results/
    #     Mistral-7B-v0.1_MDL_{ratio_name}/
    #       {method}/
    #         log.txt
    #         zero_shot_*.json   (full runs with --eval_zero_shot)
    if [ "$COMPLETION_SIGNAL" = "zero_shot" ]; then
      # zero_shot_*.json is written by main_mistral.py only after
      # eval_zero_shot fully completes — reliable end-of-job signal
      # for full runs with --eval_zero_shot passed.
      COMPLETED=$(find "$RESULTS_DIR" -maxdepth 2 -name "zero_shot_*.json" | wc -l)
    else
      # "[norms] Saved after-pruning norms" is the last line main_mistral.py
      # prints before exiting when --track_norms is passed — reliable for
      # both dry runs (no eval) and full runs.
      COMPLETED=$(grep -rl "\[norms\] Saved after-pruning norms" "$RESULTS_DIR" \
                  --include="log.txt" 2>/dev/null | wc -l)
    fi

    PERCENT=$(( COMPLETED * 100 / TOTAL_JOBS ))

    # Sleep interval shrinks as we get closer to completion —
    # early jobs take longest (model load + pruning + eval),
    # so check less frequently at the start.
    SLEEP_MIN=1  # default sleep time in minutes
    # if [ "$COMPLETED" -lt 3 ]; then
    #   SLEEP_MIN=30
    # elif [ "$COMPLETED" -lt 9 ]; then
    #   SLEEP_MIN=20
    # elif [ "$COMPLETED" -lt 18 ]; then
    #   SLEEP_MIN=15
    # elif [ "$COMPLETED" -lt 27 ]; then
    #   SLEEP_MIN=10
    # elif [ "$COMPLETED" -lt 33 ]; then
    #   SLEEP_MIN=5
    # else
    #   SLEEP_MIN=2
    # fi

    if [ "$COMPLETED" -lt 1 ]; then
      SLEEP_MIN=10
    elif [ "$COMPLETED" -lt 2 ]; then
      SLEEP_MIN=5
    elif [ "$COMPLETED" -lt 3 ]; then
      SLEEP_MIN=1
    else
      SLEEP_MIN=5
    fi

    SLEEP_SEC=$((SLEEP_MIN * 60))

    echo "  [$(date '+%Y-%m-%d %H:%M:%S')] Progress: $COMPLETED/$TOTAL_JOBS jobs done (${PERCENT}%) — next check in ${SLEEP_MIN} min..."
    sleep $SLEEP_SEC

  done

  echo "$CURRENT_SCRIPT finished. Pausing 30s for VRAM to clear..."
  sleep 30
fi

echo "=================================================="
echo "LAUNCHING: $(basename $NEXT_SCRIPT)"
echo "=================================================="
bash "$NEXT_SCRIPT"

EXIT_CODE=$?
filename=$(basename "$NEXT_SCRIPT")
if [ $EXIT_CODE -eq 0 ]; then
  echo "All $filename jobs completed successfully."
else
  echo "$filename run exited with code $EXIT_CODE — check individual log.txt files in depth_prior_results/."
fi