#!/bin/bash

# 1. Setup Paths
root_data_path="/data/mdl-layerIF/Expert_Allocation"
evaluation_script="evaluation.py"

# --- SILENCE PROGRESS BARS ---
# This environment variable disables tqdm bars in evaluation.py
export TQDM_DISABLE=1

# --- DEFINE TEST DATASETS EXPLICITLY (Like in eval_all.sh) ---
test_mrpc="$root_data_path/datasets/glue_mrpc_test.json"
test_cola="$root_data_path/datasets/glue_cola_test.json"
test_scienceq="$root_data_path/datasets/qa_text_scienceq_test_all.json"
test_commonq="$root_data_path/datasets/qa_commonq_test_all.json"
test_openbook="$root_data_path/datasets/qa_openbook_test_all.json"

# Define the training datasets to process
data_paths=(
  "$root_data_path/datasets/glue_mrpc_all.hf/"
  "$root_data_path/datasets/glue_cola_all.hf/"
  "$root_data_path/datasets/qa_text_scienceq_all.hf/"
  "$root_data_path/datasets/qa_commonq_all.hf/"
  "$root_data_path/datasets/qa_openbook_all.hf/"
)

# 2. Main Loop
for data_path in "${data_paths[@]}"; do
  
  # Clean filename (e.g., "glue_mrpc_all")
  filename=$(basename "$data_path")
  filename=${filename%/}
  filename=${filename%.hf}

  echo "=================================================="
  echo "PREPARING EVALUATION FOR: $filename"
  echo "=================================================="

  # --- 3. SELECT CORRECT TEST FILE & EXPERT CONFIG ---
  # We use the filename to pick the matching test_data variable defined above
  
  case "$filename" in
    *"commonq"*)
      current_test_file="$test_commonq"
      current_experts="1,5,6,5,4,5,5,6,7,7,7,6,5,5,7,8,7,6,8,6,7,7,6,5,5,5,4,5"
      current_top_k="1,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2"
      ;;
    *"mrpc"*)
      current_test_file="$test_mrpc"
      current_experts="1,6,6,6,6,6,5,6,5,7,8,7,7,6,6,7,6,6,7,7,6,6,5,5,4,5,3,5"
      current_top_k="1,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2"
      ;;
    *"cola"*)
      current_test_file="$test_cola"
      current_experts="1,4,7,7,8,7,8,8,8,7,7,7,7,5,6,7,6,7,5,5,5,5,5,5,4,3,3,3"
      current_top_k="1,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2"
      ;;
    *"openbook"*)
      current_test_file="$test_openbook"
      current_experts="1,5,7,6,5,6,5,8,8,8,7,7,5,4,5,7,6,4,7,7,6,7,7,3,4,6,4,5"
      current_top_k="1,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2"
      ;;
    *"scienceq"*)
      current_test_file="$test_scienceq"
      current_experts="1,12,7,7,5,5,4,5,6,8,11,5,4,4,3,7,3,3,6,6,6,6,6,7,5,5,3,10" 
      current_top_k="1,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2"
      ;;
    *)
      echo "Using default config for $filename"
      current_test_file="$test_mrpc" # Default fallback
      current_experts="1,5,7,6,5,6,5,8,8,8,7,7,5,4,5,7,6,4,7,7,6,7,7,3,4,6,4,5"
      current_top_k="1,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2"
      ;;
  esac

  # Verify test file exists
  if [ ! -f "$current_test_file" ]; then
    echo "WARNING: Test file not found: $current_test_file. Skipping..."
    continue
  fi

  # --- 4. CHECKPOINT DISCOVERY ---
  training_dir="$root_data_path/training/gemma_IF_$filename"
  
  if [ ! -d "$training_dir" ]; then
      echo "Training directory not found: $training_dir"
      continue
  fi

  # Find all checkpoint folders
  checkpoints=$(find "$training_dir" -maxdepth 1 -type d -name "checkpoint-*" | sort -V)

  if [ -z "$checkpoints" ]; then
      echo "WARNING: No checkpoints found in $training_dir."
      continue
  fi

  # --- 5. RUN EVALUATION ---
  for ckpt_path in $checkpoints; do
      ckpt_name=$(basename "$ckpt_path")

      # ------------------------------------------------------------------
      # SMART REPAIR
      # ------------------------------------------------------------------
      
      # Only try to repair if the config is MISSING (This protects checkpoint-10)
      if [ ! -f "$ckpt_path/adapter_config.json" ]; then
          
          # Repair Config
          if [ -f "$training_dir/adapter_config.json" ]; then
             echo "   [REPAIR] Missing config in $ckpt_name. Copying from parent..."
             cp "$training_dir/adapter_config.json" "$ckpt_path/"
          elif [ -f "$training_dir/checkpoint-10/adapter_config.json" ]; then
             echo "   [REPAIR] Missing config in $ckpt_name. Copying from checkpoint-10..."
             cp "$training_dir/checkpoint-10/adapter_config.json" "$ckpt_path/"
          fi

          # Repair Weights (Rename pytorch_model.bin -> adapter_model.bin)
          if [ ! -f "$ckpt_path/adapter_model.bin" ] && [ -f "$ckpt_path/pytorch_model.bin" ]; then
              echo "   [REPAIR] Renaming weights in $ckpt_name..."
              cp "$ckpt_path/pytorch_model.bin" "$ckpt_path/adapter_model.bin"
          fi
      fi
      # ------------------------------------------------------------------
      
      echo "   Evaluating Checkpoint: $ckpt_name"
      echo "   Test Data: $current_test_file"
      
      save_json="$training_dir/eval_results_${ckpt_name}.json"
      eval_log="$training_dir/eval_console_${ckpt_name}.log"

      echo "   Running: $ckpt_name (Logging to: $(basename "$eval_log"))"

      # Launch Evaluation
      # We rely on 'device_map="auto"' inside evaluation.py to distribute across 4 GPUs
      CUDA_VISIBLE_DEVICES=2,3 python "$evaluation_script" \
        --test_dataset "$current_test_file" \
        --base_model "google/gemma-7b" \
        --mola_weights "$ckpt_path" \
        --number_experts "$current_experts" \
        --top_k "$current_top_k" \
        --lora_target_modules "q_proj,v_proj,k_proj,o_proj,gate_proj,down_proj,up_proj" \
        --batch_size 8 \
        --save_path "$save_json" \
        > "$eval_log" 2>&1

      if [ $? -eq 0 ]; then
         echo "   [SUCCESS] Saved to $save_json"
      else
         echo "   [FAILURE] Error evaluating $eval_log"
      fi

  done

done