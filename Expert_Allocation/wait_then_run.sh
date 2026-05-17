#!/bin/bash

CURRENT_SCRIPT="run_all_mola_gemma_sequential.sh" # script that is running currently
# script that we want to run after waiting for the above to finish:
NEXT_SCRIPT="/data/mdl-layerIF/Expert_Allocation/run_all_alpha_gemma_sequential.sh"

echo "=================================================="
echo "Searching for running job ($CURRENT_SCRIPT)..."
echo "=================================================="

CURRENT_PID=$(pgrep -f "$CURRENT_SCRIPT" | head -1)

if [ -z "$CURRENT_PID" ]; then
  echo "WARNING: No running $CURRENT_SCRIPT job found. Launching AlphaLora immediately..."
else
  echo "Found $CURRENT_SCRIPT job with PID: $CURRENT_PID. Waiting for it to finish..."
  while kill -0 "$CURRENT_PID" 2>/dev/null; do
    echo "  [$(date '+%Y-%m-%d %H:%M:%S')] $CURRENT_SCRIPT still running (PID $CURRENT_PID) — sleeping 60s..."
    sleep 60
  done
  echo " $CURRENT_SCRIPT finished. Pausing 30s for VRAM to clear..."
  sleep 30
fi

echo "=================================================="
echo "LAUNCHING AlphaLora sequential experiment"
echo "=================================================="
bash "$NEXT_SCRIPT"

EXIT_CODE=$?
filename=$(basename "$NEXT_SCRIPT")
if [ $EXIT_CODE -eq 0 ]; then
    
  echo "All $filename jobs completed successfully."
else
  echo " $filename run exited with code $EXIT_CODE — check individual training.log files."
fi