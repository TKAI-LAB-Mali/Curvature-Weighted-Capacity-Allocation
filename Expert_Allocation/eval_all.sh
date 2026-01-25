#!/bin/bash

# Set base paths for data and weights

base_weights_path=""/nas02/Hadi/Model-Selection-IF/alphalora""

#The weights path should be the same with the output_dir in run_all.sh
mola_weights=(
  # "$base_weights_path/mistral_mola_10ep_2b_epoch_glue_rte_all"
  "$base_weights_path/gemma_160_mola_glue_mrpc_all"
  "$base_weights_path/gemma_160_mola_glue_cola_all"
  "$base_weights_path/gemma_160_mola_qa_text_scienceq_all"
  "$base_weights_path/gemma_160_mola_qa_commonq_all"
  "$base_weights_path/gemma_160_mola_qa_openbook_all"

)

test_data=(
  # "$base_weights_path/datasets/glue_rte_test.json"
  "$base_weights_path/datasets/glue_mrpc_test.json"
  "$base_weights_path/datasets/glue_cola_test.json"
  "$base_weights_path/datasets/qa_text_scienceq_test_all.json"
  "$base_weights_path/datasets/qa_commonq_test_all.json"
  "$base_weights_path/datasets/qa_openbook_test_all.json"
)

checkpoints=("checkpoint-198" "checkpoint-84" "checkpoint-228" "checkpoint-114" "checkpoint-150")

# Loop through the arrays and run evaluations
#

for checkpoint in "${checkpoints[@]}"; do
  for (( i=0; i<${#mola_weights[@]}; i++ )); do
    if [[ $checkpoint == "checkpoint-20" ]]; then
      checkpoint_dir="${mola_weights[i]}"
    else
      checkpoint_dir="${mola_weights[i]}/${checkpoint}"
    fi

    if [ -d "$checkpoint_dir" ]; then
      echo "Evaluating $(basename ${mola_weights[i]}) - $checkpoint"
      CUDA_VISIBLE_DEVICES=0,1,2,3 python evaluation.py \
        --test_dataset "${test_data[i]}" \
        --base_model "google/gemma-7b" \
        --mola_weights "$checkpoint_dir" \
        --batch_size 8 \
        --lora_target_modules "q_proj,v_proj,k_proj,o_proj,gate_proj,down_proj,up_proj" \
        --number_experts "3,3,3,3,3,3,3,5,5,5,5,5,5,5,7,7,7,7,7,7,7,7,8,8,8,8,8,8" \
        --top_k "2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2" \
        --save_path "$base_weights_path/results/gemma_mola_160/$(basename ${mola_weights[i]})_${checkpoint}.json"
    else
      echo "Checkpoint directory $checkpoint_dir does not exist."
    fi
  done
done