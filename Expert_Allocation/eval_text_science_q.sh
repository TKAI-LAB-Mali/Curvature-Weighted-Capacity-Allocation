#!/bin/bash

# Set base paths for data and weights

base_weights_path="/nas02/Hadi/Model-Selection-IF/alphalora"

#The weights path should be the same with the output_dir in run_all.sh
mola_weights=(
  # "$base_weights_path/mistral_IF_negative_3_glue_rte_all"
  # "$base_weights_path/mistral_IF_negative_3_glue_mrpc_all"
  # "$base_weights_path/mistral_IF_negative_3_glue_cola_all"
  "$base_weights_path/gemma_IF_3_160_qa_text_scienceq_all"
  # "$base_weights_path/mistral_IF_negative_3_qa_commonq_all"
  # "$base_weights_path/mistral_IF_negative_3_qa_openbook_all"

)

test_data=(
  # "$base_weights_path/datasets/glue_rte_test.json"
  # "$base_weights_path/datasets/glue_mrpc_test.json"
  # "$base_weights_path/datasets/glue_cola_test.json"
  "$base_weights_path/datasets/qa_text_scienceq_test_all.json"
  # "$base_weights_path/datasets/qa_commonq_test_all.json"
  # "$base_weights_path/datasets/qa_openbook_test_all.json"
)

# checkpoints=("checkpoint-10" "checkpoint-15" "checkpoint-20")

checkpoints=("checkpoint-150")
# checkpoints=("checkpoint-10")

# Loop through the arrays and run evaluations
#

for (( i=0; i<${#mola_weights[@]}; i++ )); do
  for checkpoint in "${checkpoints[@]}"; do
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
        --number_experts "1,15,7,7,4,4,3,5,5,10,14,4,3,3,2,8,3,2,5,6,6,5,5,7,5,5,2,14" \
        --top_k "1,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2" \
        --save_path "$base_weights_path/results/gemma_IF_160_3/$(basename ${mola_weights[i]})_${checkpoint}.json"
    else
      echo "Checkpoint directory $checkpoint_dir does not exist."
    fi
  done
done
