#!/bin/bash

# 1. Setup Paths
root_data_path="/data/mdl-layerIF/Expert_Allocation"
evaluation_script="evaluation.py"

# --- CONFIGURATION ---
export TQDM_DISABLE=1  # Silence progress bars

# --- DEFINE TEST DATASETS ---
test_mrpc="$root_data_path/datasets/glue_mrpc_test.json"
test_cola="$root_data_path/datasets/glue_cola_test.json"
test_scienceq="$root_data_path/datasets/qa_text_scienceq_test_all.json"
test_commonq="$root_data_path/datasets/qa_commonq_test_all.json"
test_openbook="$root_data_path/datasets/qa_openbook_test_all.json"

# Define the training datasets to process
data_paths=(
  "$root_data_path/datasets/glue_mrpc_all.hf/"
  "$root_data_path/datasets/glue_cola_all.hf/"
#   "$root_data_path/datasets/qa_text_scienceq_all.hf/"
#   "$root_data_path/datasets/qa_commonq_all.hf/"
#   "$root_data_path/datasets/qa_openbook_all.hf/"
)
Set1="Our_positive_IF"
Set2="Have_positive_IF"
sub_directory="positive_IF"

# Define the Experiment Sets to Evaluate
experiment_sets=("$Set1" "$Set2")

# 2. Main Loop
for data_path in "${data_paths[@]}"; do
  
  filename=$(basename "$data_path")
  filename=${filename%/}
  filename=${filename%.hf}

  echo "=================================================="
  echo "PROCESSING DATASET: $filename"
  echo "=================================================="

  # Determine Test File
  case "$filename" in
    *"commonq"*)  current_test_file="$test_commonq" ;;
    *"mrpc"*)     current_test_file="$test_mrpc" ;;
    *"cola"*)     current_test_file="$test_cola" ;;
    *"openbook"*) current_test_file="$test_openbook" ;;
    *"scienceq"*) current_test_file="$test_scienceq" ;;
    *)            current_test_file="$test_mrpc" ;;
  esac

  if [ ! -f "$current_test_file" ]; then
    echo "WARNING: Test file not found: $current_test_file. Skipping..."
    continue
  fi

  # --- INNER LOOP: PROCESS BOTH SETS (Set1 & Set2) ---
  for run_suffix in "${experiment_sets[@]}"; do
      
      echo "   >>> Evaluating Run: $run_suffix"

      # 3. DEFINE CONFIGURATION BASED ON SET
      # (These must match what you used in run_all_gemma_sequential_v2.sh)
      
      if [ "$run_suffix" == "$Set1" ]; then
          # --- SET 1 CONFIGS (Original) ---
          case "$filename" in
            *"commonq"*)
            current_experts="5,5,5,5,5,5,5,5,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6"
            current_top_k="1,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2"
            ;;
            *"mrpc"*)
              current_experts="5,5,5,6,6,6,6,6,6,6,1,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6"
              current_top_k="1,2,2,2,2,2,2,2,2,2,1,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2"
              ;;
            # *"mrpc"*)
            # current_experts="1,1,1,1,1,1,1,1,1,1,133,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1"
            # current_top_k="1,1,1,1,1,1,1,1,1,1,2,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1"
            # ;;
            # *"cola"*)
            # current_experts="1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,133,1,1,1"
            # current_top_k="1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,2,1,1,1"
            # ;;
            *"cola"*)
              current_experts="5,5,5,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,1,6,6,6"
              current_top_k="1,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,1,2,2,2"
              ;;
            *"openbook"*)
            current_experts="3,1,6,3,6,6,6,6,6,6,6,6,6,6,6,6,6,7,6,7,7,7,7,6,7,1,7,7"
            current_top_k="1,1,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,1,2,2"
            ;;
            *"scienceq"*)
            current_experts="1,6,6,6,6,6,6,6,6,6,5,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,5,5" 
            current_top_k="1,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2"
            ;;
            *)
            echo "Using default config for $filename"
            current_experts="1,5,7,6,5,6,5,8,8,8,7,7,5,4,5,7,6,4,7,7,6,7,7,3,4,6,4,5"
            current_top_k="1,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2"
            ;;
          esac

      elif [ "$run_suffix" == "$Set2" ]; then
          # --- SET 2 CONFIGS (New Experiments) ---
          case "$filename" in
            # *"mrpc"*)
            # current_experts="1,5,7,6,6,6,5,6,4,8,9,8,8,6,6,7,6,5,7,7,6,5,5,5,4,5,2,5"
            # current_top_k="1,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2"
            # ;;
            # *"cola"*)
            # current_experts="1,3,7,7,8,8,9,9,9,8,7,8,8,5,6,8,6,7,5,5,4,4,4,4,3,2,2,3"
            # current_top_k="1,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2"
            # ;;

            *"commonq"*)
            current_experts="1,5,6,5,4,5,4,6,7,8,7,6,4,4,7,10,7,6,9,6,7,7,6,5,5,5,3,5"
            current_top_k="1,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2"
            ;;
            *"mrpc"*)
            current_experts="1,5,7,6,6,6,5,6,4,8,9,8,8,6,6,7,6,5,7,7,6,5,5,5,4,5,2,5"
            current_top_k="1,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2"
            ;;
            *"cola"*)
            current_experts="1,3,7,7,8,8,9,9,9,8,7,8,8,5,6,8,6,7,5,5,4,4,4,4,3,2,2,3"
            current_top_k="1,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2"
            ;;
            *"openbook"*)
            current_experts="1,5,8,6,5,6,4,8,8,10,7,8,4,3,5,7,6,3,8,8,6,7,7,2,4,6,3,5"
            current_top_k="1,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2"
            ;;
            *"scienceq"*)
            current_experts="1,17,7,7,4,5,3,5,5,10,14,4,3,3,2,7,2,2,5,6,5,5,5,7,5,5,2,14" 
            current_top_k="1,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2"
            ;;
            *)
            echo "Using default config for $filename"
            current_experts="1,5,7,6,5,6,5,8,8,8,7,7,5,4,5,7,6,4,7,7,6,7,7,3,4,6,4,5"
            current_top_k="1,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2"
            ;;
          esac
      fi

      # 4. FIND CHECKPOINTS
      training_dir="$root_data_path/training/${sub_directory}/gemma_IF_${filename}_${run_suffix}"
      
      if [ ! -d "$training_dir" ]; then
          echo "      [SKIP] Directory not found: $training_dir"
          continue
      fi

      # =========================================================
      # STEP 5a: EVALUATE FINAL MODEL (PARENT DIRECTORY)
      # =========================================================
      # This uses the adapter_model.bin sitting directly in the folder
      
      if [ -f "$training_dir/adapter_model.bin" ] && [ -f "$training_dir/adapter_config.json" ]; then
          
          save_json="$training_dir/eval_results_FINAL.json"
          eval_log="$training_dir/eval_console_FINAL.log"

          echo "      Running: FINAL MODEL (Parent Dir)"
          
          CUDA_VISIBLE_DEVICES=0,1 python "$evaluation_script" \
            --test_dataset "$current_test_file" \
            --base_model "google/gemma-7b" \
            --mola_weights "$training_dir" \
            --number_experts "$current_experts" \
            --top_k "$current_top_k" \
            --lora_target_modules "q_proj,v_proj,k_proj,o_proj,gate_proj,down_proj,up_proj" \
            --batch_size 8 \
            --save_path "$save_json" \
            > "$eval_log" 2>&1

          if [ $? -eq 0 ]; then
             echo "         [DONE] Saved to eval_results_FINAL.json"
          else
             echo "         [FAILED] Check log: $eval_log"
          fi
      else
          echo "      [INFO] No Final Model in parent directory."
      fi
      # =========================================================
      # 5b. RUN EVALUATION ON CHECKPOINTS

      # =========================================================

      checkpoints=$(find "$training_dir" -maxdepth 1 -type d -name "checkpoint-*" | sort -V)

      if [ -z "$checkpoints" ]; then
          echo "      [SKIP] No checkpoints found in $run_suffix."
          continue
      fi

      
      for ckpt_path in $checkpoints; do
          ckpt_name=$(basename "$ckpt_path")
          
          # Skip very early checkpoints if desired
          if [[ "$ckpt_name" == "checkpoint-10" ]] && [[ "$run_suffix" == "$Set1" ]]; then
              # Keep checkpoint-10 for Set1 if that's your only good one from the past
              :
          elif [[ "$ckpt_name" == "checkpoint-10" ]]; then
               # For new runs, we usually want the later checkpoints, but keeping 10 is fine too.
               :
          fi

          # --- CHECK FOR VALIDITY (Since you fixed the Trainer, these should be good) ---
          if [ ! -f "$ckpt_path/adapter_config.json" ]; then
              echo "      [WARN] Missing config in $ckpt_name. Attempting auto-repair from parent..."
              if [ -f "$training_dir/adapter_config.json" ]; then
                  cp "$training_dir/adapter_config.json" "$ckpt_path/"
              else
                  echo "      [SKIP] Cannot repair. Skipping."
                  continue
              fi
          fi

          save_json="$training_dir/eval_results_${ckpt_name}.json"
          eval_log="$training_dir/eval_console_${ckpt_name}.log"

          echo "      Running: $ckpt_name"
          
          CUDA_VISIBLE_DEVICES=0,1 python "$evaluation_script" \
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
             echo "         [DONE] Saved to eval_results_${ckpt_name}.json"
          else
             echo "         [FAILED] Check log: $eval_log"
          fi

      done # End Checkpoint Loop

  done # End Set Loop

done # End Dataset Loop

echo "All evaluations finished."