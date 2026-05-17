#!/bin/bash

CURRENT_SCRIPT="run_all_mola_gemma_sequential_eval.sh"
NEXT_SCRIPT="/data/mdl-layerIF/Expert_Allocation/run_all_alpha_gemma_sequential_eval.sh"

# Directory where mola saves output folders, and total number of datasets
MOLA_OUTPUT_DIR="/data/mdl-layerIF/Expert_Allocation/training/mola3_5_7_8_baseline"
TOTAL_DATASETS=5

echo "=================================================="
echo "Searching for running job ($CURRENT_SCRIPT)..."
echo "=================================================="

# CURRENT_PID=$(pgrep -f "$CURRENT_SCRIPT" | head -1)
CURRENT_PID=$(pgrep -o -f "$CURRENT_SCRIPT")

if [ -z "$CURRENT_PID" ]; then
  echo "WARNING: No running job found. Launching next script immediately..."
else
  echo "Found job with PID: $CURRENT_PID. Waiting for it to finish..."

  while kill -0 "$CURRENT_PID" 2>/dev/null; do

    # We use maxdepth 2: 
    # Level 1 is the dataset folder (e.g., /mola3_5_7_8_baseline/dataset_1/)
    # Level 2 is the file itself (e.g., /mola3_5_7_8_baseline/dataset_1/adapter_model.bin)
    # COMPLETED=$(find "$MOLA_OUTPUT_DIR" -maxdepth 2 -name "adapter_model.bin" | wc -l)
    COMPLETED=$(find "$MOLA_OUTPUT_DIR" -maxdepth 2 -name "eval_results_FINAL.json" | wc -l)


    # Determine sleep interval based on progress
    if [ "$COMPLETED" -eq 0 ]; then
      SLEEP_MIN=30
    elif [ "$COMPLETED" -eq 1 ]; then
      SLEEP_MIN=20
    elif [ "$COMPLETED" -eq 2 ]; then
      SLEEP_MIN=15
    elif [ "$COMPLETED" -eq 3 ]; then
      SLEEP_MIN=10
    elif [ "$COMPLETED" -eq 4 ]; then
      SLEEP_MIN=5
    else
      SLEEP_MIN=1
    fi

    SLEEP_SEC=$((SLEEP_MIN * 60))

    echo "  [$(date '+%Y-%m-%d %H:%M:%S')] Progress: $COMPLETED/$TOTAL_DATASETS datasets done — next check in ${SLEEP_MIN} min..."
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
  echo "$filename run exited with code $EXIT_CODE — check individual training.log files."
fi