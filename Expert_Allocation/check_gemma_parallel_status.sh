#!/bin/bash

# Clear the screen for a fresh dashboard view
clear

echo "=========================================================="
echo "                 GPU STATUS (NVIDIA-SMI)                  "
echo "=========================================================="
# Show GPU memory and utilization
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader

echo ""
echo "=========================================================="
echo "            ACTIVE TRAINING JOBS (PYTHON)                 "
echo "=========================================================="

# List running python processes matching your script name
# We use 'ps' to format the output nicely:
# - pid: Process ID
# - start_time: When it started
# - cmd: The full command (so you can see which dataset is running)
ps -eo pid,stime,cmd | grep "mola_training_gemma.py" | grep -v "grep" | while read -r line; do
    
    # Extract just the dataset path argument to make it readable
    pid=$(echo "$line" | awk '{print $1}')
    stime=$(echo "$line" | awk '{print $2}')
    
    # Find the chunk of text that looks like a dataset path (contains 'datasets/')
    dataset=$(echo "$line" | grep -o 'datasets/[^ ]*' | xargs basename)
    
    # If we couldn't find the dataset name easily, just print a generic label
    if [ -z "$dataset" ]; then
        dataset="Unknown Dataset (Check full logs)"
    fi

    echo "▶ [PID: $pid] Started: $stime | Training: $dataset"
done

# Check if nothing was found
if ! pgrep -f "mola_training_gemma.py" > /dev/null; then
    echo "No training jobs found. Is the script running?"
fi

echo ""
echo "=========================================================="