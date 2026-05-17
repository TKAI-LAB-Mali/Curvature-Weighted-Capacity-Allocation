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
  "$root_data_path/datasets/qa_text_scienceq_all.hf/"
  "$root_data_path/datasets/qa_commonq_all.hf/"
  "$root_data_path/datasets/qa_openbook_all.hf/"
)


sub_directory="positive_mdl_IF_correct"
run_suffix="positive_mdl_IF"



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


      
  echo "   >>> Evaluating Run: $run_suffix"

      # 3. DEFINE CONFIGURATION BASED ON SET
  case "$filename" in
  *"commonq"*)
  current_experts="1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1"
  current_top_k="1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1"
  ;;
  *"mrpc"*)
  current_experts="5,1,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5"
  current_top_k="2,1,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2"
  ;;
  *"cola"*)
  current_experts="5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,1,5,5,5,5,5,5,5,5,5,5,5"
  current_top_k="2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,1,2,2,2,2,2,2,2,2,2,2,2"
  ;;     
  *"openbook"*)
  current_experts="3,6,6,6,6,6,6,6,6,6,6,1,6,6,6,6,6,1,6,6,6,3,6,6,6,6,6,6"
  current_top_k="2,2,2,2,2,2,2,2,2,2,2,1,2,2,2,2,2,1,2,2,2,2,2,2,2,2,2,2"
  ;;
  *"scienceq"*)
  current_experts="1,5,5,6,5,5,5,5,6,6,6,5,6,5,5,5,5,5,5,5,6,5,5,5,5,5,6,5" 
  current_top_k="1,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2"
  ;;
  *)
  echo "Using default config for $filename"
  current_experts="1,5,7,6,5,6,5,8,8,8,7,7,5,4,5,7,6,4,7,7,6,7,7,3,4,6,4,5"
  current_top_k="1,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2"
  ;;
  esac
      
      

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
          
          CUDA_VISIBLE_DEVICES=0,1,2,3 python "$evaluation_script" \
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
          
          CUDA_VISIBLE_DEVICES=0,1,2,3 python "$evaluation_script" \
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



done # End Dataset Loop

echo "All evaluations finished."