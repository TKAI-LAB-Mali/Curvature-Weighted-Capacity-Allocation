#!/bin/bash

root_data_path="/data/mdl-layerIF/Expert_Allocation"

# 1. Define Datasets
data_paths=(
  "$root_data_path/datasets/glue_mrpc_all.hf/"
  "$root_data_path/datasets/glue_cola_all.hf/"
  "$root_data_path/datasets/qa_text_scienceq_all.hf/"
  "$root_data_path/datasets/qa_commonq_all.hf/"
  "$root_data_path/datasets/qa_openbook_all.hf/"
)

sub_directory="mdl_IF_beta_3_mbatch_4"
Set1="All_mdl_IF"
Set2="Hadi_positive_IF"

# --- FUNCTION: Handles the training execution ---
# Accepts: Dataset Path, Experts List, Top K, Run Suffix
run_training() {
    local data_path=$1
    local current_experts=$2
    local current_top_k=$3
    local run_suffix=$4  # e.g., "Set1" or "NewConfig"

    # Clean filename
    local filename=$(basename "$data_path")
    filename=${filename%/}
    filename=${filename%.hf}
    
    # CRITICAL: Append suffix to output_dir to prevent overwriting!
    local output_dir="$root_data_path/training/${sub_directory}/gemma_IF_${filename}_${run_suffix}"
    mkdir -p -m 770 "$output_dir"
    local log_file="$output_dir/training.log"
    
    # Generate random port
    local port=$((29500 + $RANDOM % 100))

    echo "--------------------------------------------------"
    echo "STARTING: $filename ($run_suffix)"
    echo "OUTPUT: $output_dir"
    echo "--------------------------------------------------"

    torchrun --nproc_per_node=4 --master_port=$port mola_training_gemma.py \
      --base_model "google/gemma-7b" \
      --data_path "$data_path" \
      --output_dir "$output_dir" \
      --batch_size 128 \
      --micro_batch_size 4 \
      --num_epochs 5 \
      --learning_rate 3e-4 \
      --cutoff_len 256 \
      --val_set_size 1 \
      --lora_r "8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8" \
      --lora_alpha 16 \
      --lora_dropout 0.05 \
      --lora_target_modules "q_proj,v_proj,k_proj,o_proj,gate_proj,down_proj,up_proj" \
      --number_experts "$current_experts" \
      --top_k "$current_top_k" \
      --train_on_inputs \
      --group_by_length \
      --add_eos_token \
      --wandb_project "IF Layer Quality" \
      --wandb_run_name "${sub_directory}_gemma_IF_${filename}_${run_suffix}" \
      > "$log_file" 2>&1

    if [ $? -eq 0 ]; then
      echo "SUCCESS: $filename ($run_suffix) finished."
    else
      echo "FAILURE: $filename ($run_suffix) crashed. Check $log_file"
    fi

    # Cool down
    sleep 10
}

# ==========================================
# EXPERIMENT SET 1 (Original)
# ==========================================
echo "### STARTING EXPERIMENT SET 1 ###"

for data_path in "${data_paths[@]}"; do
  filename=$(basename "$data_path" .hf)
  
  case "$filename" in
    #===========================================
    # Beta = 2
    #===========================================
    #   *"mrpc"*)
    #     current_experts="1,8,7,7,6,6,7,6,6,7,7,6,6,6,5,5,4,5,2,5,6,6,6,6,6,6,5,7"
    #     current_top_k="1,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2"
    #     ;; 
    #   *"cola"*)
    #     current_experts="1,7,7,7,6,6,7,6,7,5,5,4,5,5,5,5,4,3,3,3,7,7,8,7,8,8,8,7"
    #     current_top_k="1,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2"
    #     ;; 
    #   *"openbook"*)
    #     current_experts="1,7,7,5,5,5,7,6,5,7,7,5,6,6,6,4,5,6,4,5,7,6,5,6,5,7,7,8"
    #     current_top_k="1,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2"
    #     ;; 
    #   *"scienceq"*)
    #     current_experts="1,9,5,5,4,4,7,4,4,6,6,10,6,6,6,7,6,5,3,9,7,7,5,6,5,6,6,8"
    #     current_top_k="1,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2"
    #     ;; 
    #   *"commonq"*)
    #     current_experts="1,7,6,5,5,7,9,7,6,8,6,5,7,7,6,5,5,5,3,5,6,5,4,5,4,6,7,7"
    #     current_top_k="1,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2"
    #     ;;
    #===========================================
    # Beta = 3
    #===========================================
        *"mrpc"*)
        current_experts="1,9,8,8,6,6,7,6,5,7,7,5,7,5,5,5,3,4,1,5,7,7,6,6,5,6,4,8"
        current_top_k="1,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,1,2,2,2,2,2,2,2,2,2"
        ;; 
        *"cola"*)
        current_experts="1,7,8,8,5,6,8,6,7,5,5,3,4,4,4,4,3,2,2,2,7,8,9,8,9,9,9,8"
        current_top_k="1,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2"
        ;; 
        *"openbook"*)
        current_experts="1,7,7,5,4,5,7,6,4,7,7,5,6,7,7,3,4,6,3,5,7,6,5,6,5,8,8,9"
        current_top_k="1,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2"
        ;; 
        *"scienceq"*)
        current_experts="1,11,5,4,4,3,7,3,3,6,6,12,6,6,6,7,5,5,2,11,7,7,5,5,4,5,6,9"
        current_top_k="1,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2"
        ;; 
        *"commonq"*)
        current_experts="1,8,6,4,4,8,11,7,6,9,6,4,7,7,6,5,5,5,3,5,6,5,3,5,4,6,7,8"
        current_top_k="1,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2"
        ;; 
      *)
      echo "Using default config for $filename"
      current_experts="1,5,7,6,5,6,5,8,8,8,7,7,5,4,5,7,6,4,7,7,6,7,7,3,4,6,4,5"
      current_top_k="1,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2"
      ;;
  esac

  # Call Function with "Set1" tag
  run_training "$data_path" "$current_experts" "$current_top_k" "$Set1"

done

# ==========================================
# EXPERIMENT SET 2 (New)
# ==========================================
# echo "### STARTING EXPERIMENT SET 2 ###"

# for data_path in "${data_paths[@]}"; do
#   filename=$(basename "$data_path" .hf)

#   # --- PASTE YOUR NEW EXPERT CONFIGURATIONS HERE ---
#   case "$filename" in
#     *"commonq"*)
#       current_experts="1,5,6,5,4,5,4,6,7,8,7,6,4,4,7,10,7,6,9,6,7,7,6,5,5,5,3,5"
#       current_top_k="1,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2"
#       ;;
#     *"mrpc"*)
#       current_experts="1,5,7,6,6,6,5,6,4,8,9,8,8,6,6,7,6,5,7,7,6,5,5,5,4,5,2,5"
#       current_top_k="1,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2"
#       ;;
#     *"cola"*)
#       current_experts="1,3,7,7,8,8,9,9,9,8,7,8,8,5,6,8,6,7,5,5,4,4,4,4,3,2,2,3"
#       current_top_k="1,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2"
#       ;;
#     *"openbook"*)
#       current_experts="1,5,8,6,5,6,4,8,8,10,7,8,4,3,5,7,6,3,8,8,6,7,7,2,4,6,3,5"
#       current_top_k="1,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2"
#       ;;
#     *"scienceq"*)
#       current_experts="1,17,7,7,4,5,3,5,5,10,14,4,3,3,2,7,2,2,5,6,5,5,5,7,5,5,2,14" 
#       current_top_k="1,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2"
#       ;;
#     *)
#       echo "Using default config for $filename"
#       current_experts="1,5,7,6,5,6,5,8,8,8,7,7,5,4,5,7,6,4,7,7,6,7,7,3,4,6,4,5"
#       current_top_k="1,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2"
#       ;;
#   esac

  # Call Function with "Set2" tag (Outputs to: ..._Set2/)
#   run_training "$data_path" "$current_experts" "$current_top_k" "$Set2"

# done

echo "All jobs finished."