#!/bin/bash

# Set base paths for data and weights

base_weights_path="/data/mdl-layerIF/Expert_Allocation"

#The weights path should be the same with the output_dir in run_all.sh
mola_weights=(
  # "$base_weights_path/mistral_IF_negative_3_glue_rte_all"
  "$base_weights_path/results/all-negative/mdl-output/mistral_IF_5_epoch_glue_mrpc_all"
  "$base_weights_path/results/all-negative/mdl-output/mistral_IF_5_epoch_glue_cola_all"
  "$base_weights_path/results/all-negative/mdl-output/mistral_IF_5_epoch_qa_text_scienceq_all"
  "$base_weights_path/results/all-negative/mdl-output/mistral_IF_5_epoch_qa_commonq_all"
  "$base_weights_path/results/all-negative/mdl-output/mistral_IF_5_epoch_qa_openbook_all"

)

test_data=(
  # "$base_weights_path/datasets/glue_rte_test.json"
  "$base_weights_path/datasets/glue_mrpc_test.json"
  "$base_weights_path/datasets/glue_cola_test.json"
  "$base_weights_path/datasets/qa_text_scienceq_test_all.json"
  "$base_weights_path/datasets/qa_commonq_test_all.json"
  "$base_weights_path/datasets/qa_openbook_test_all.json"
)

checkpoints=("checkpoint-335" "checkpoint-255" "checkpoint-195" "checkpoint-385" "checkpoint-145")

# Loop through the arrays and run evaluations
#
get_expert_config_negative_layerIFs() {
  local dataset=$1
  case $dataset in
    *mrpc*)
      number_experts="1,6,8,5,6,11,10,10,10,10,8,10,6,7,6,4,6,4,3,1,5,5,2,1,5,2,1,3,1,1,1,1"
      top_k="1,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,1,2,2,2,1,2,2,1,2,1,1,1,1"
      ;;
    *cola*)
      number_experts="1,14,10,7,9,11,10,7,8,8,9,7,5,9,6,3,7,3,3,2,3,4,2,1,4,1,1,1,1,1,1,1"
      top_k="1,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,1,2,1,1,1,1,1,1,1"
      ;;
    *scienceq*|*text_science*)
      number_experts="1,1,2,1,2,2,4,5,4,6,6,6,6,6,7,6,6,6,6,7,7,6,6,5,5,5,7,6,5,6,7,5"
      top_k="1,1,2,1,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2"
      ;;
    *commonq*)
      number_experts="1,8,6,8,7,5,10,6,13,9,8,9,6,8,7,3,7,6,5,2,5,4,2,1,2,3,2,2,1,1,2,1"
      top_k="1,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,1,2,2,2,2,1,1,2,1"
      ;;
    *openbook*)
      number_experts="1,5,4,8,7,5,8,7,7,9,7,7,6,6,6,6,6,6,7,5,6,5,3,2,2,4,3,4,2,2,3,1"
      top_k="1,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,1"
      ;;
    *rte*)
      number_experts="1,1,3,5,4,5,8,8,8,8,9,8,8,7,7,7,6,7,6,4,4,7,4,3,6,4,2,3,1,3,2,1"
      top_k="1,1,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,1,2,2,2,2,2,2,2,2,1,2,2,1"
      ;;
    *)
      # Default configuration
      number_experts="1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1"
      top_k="1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1"
      echo "Warning: No specific config for $dataset, using default"
      ;;
  esac
}

get_expert_config_negative_IFs_mdl() {
  local dataset=$1
  case $dataset in
    *mrpc*) # rho - 0.0277
      number_experts="3,6,7,7,6,6,6,5,6,5,5,3,7,5,5,3,2,5,4,3,4,2,3,6,2,2,6,7,7,7,7,7"
      top_k="2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2"
      ;;
    *cola*) # rho - 0.0245
      number_experts="3,8,6,6,5,6,5,4,6,4,4,3,6,4,4,4,2,5,2,1,3,1,3,6,2,1,11,7,11,6,10,10"
      top_k="2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,1,2,1,2,2,2,1,2,2,2,2,2,2"
      ;;
    *scienceq*|*text_science*) # rho - 0.0276
      number_experts="1,2,6,6,6,6,6,6,6,6,6,6,3,6,6,6,5,5,5,6,6,5,6,1,6,5,3,3,4,5,4,6"
      top_k="1,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,1,2,2,2,2,2,2,2,2"
      ;;
    *commonq*) # rho - 0.0276
      number_experts="2,6,6,7,6,6,6,5,6,6,5,4,6,5,5,4,2,4,5,4,4,2,3,6,4,3,6,5,7,6,7,6"
      top_k="2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2"
      ;;
    *openbook*) # rho - 0.028
      number_experts="1,5,6,6,6,6,6,6,6,6,6,5,5,6,5,4,3,3,5,4,4,3,4,6,4,2,6,5,7,6,6,7"
      top_k="1,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2"
      ;;
    *rte*)
      number_experts="1,1,3,5,4,5,8,8,8,8,9,8,8,7,7,7,6,7,6,4,4,7,4,3,6,4,2,3,1,3,2,1"
      top_k="1,1,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,1,2,2,2,2,2,2,2,2,1,2,2,1"
      ;;
    *)
      # Default configuration
      number_experts="1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1"
      top_k="1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1"
      echo "Warning: No specific config for $dataset, using default"
      ;;
  esac
}

for (( i=0; i<${#mola_weights[@]}; i++ )); do
  for checkpoint in "${checkpoints[@]}"; do
    # if [[ $checkpoint == "checkpoint-20" ]]; then
    checkpoint_dir="${mola_weights[i]}"
    # else
    # checkpoint_dir="${mola_weights[i]}/${checkpoint}"
    # fi

    if [ -d "$checkpoint_dir" ]; then
      get_expert_config_negative_IFs_mdl "${test_data[i]}"

      echo "Evaluating $(basename ${mola_weights[i]})"
      CUDA_VISIBLE_DEVICES=1 python evaluation.py \
        --test_dataset "${test_data[i]}" \
        --base_model "mistralai/Mistral-7B-v0.1" \
        --mola_weights "$checkpoint_dir" \
        --batch_size 8 \
        --lora_target_modules "q_proj,v_proj,k_proj,o_proj,gate_proj,down_proj,up_proj" \
        --number_experts "$number_experts" \
        --top_k "$top_k" \
        --save_path "$base_weights_path/results/all-negative/mdl-output/$(basename ${mola_weights[i]})/eval_result.json"
    # else
    #   echo "Checkpoint directory $checkpoint_dir does not exist."
    fi
  done
done