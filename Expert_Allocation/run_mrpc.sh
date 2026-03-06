#!/bin/bash

# Set the root data path
root_data_path="/nas02/Hadi/Model-Selection-IF/alphalora"

# Define data paths using the root path
data_paths=(
  # "$root_data_path/datasets/glue_rte_all.hf/"
  "$root_data_path/datasets/glue_mrpc_all.hf/"
  # "$root_data_path/datasets/glue_cola_all.hf/"
  # "$root_data_path/datasets/qa_text_scienceq_all.hf/"
  # "$root_data_path/datasets/qa_commonq_all.hf/"
  # "$root_data_path/datasets/qa_openbook_all.hf/"
)

# Loop through data paths and run experiments
for data_path in "${data_paths[@]}"; do
  # Extract filename from data path
  filename=$(basename "$data_path")
  # Remove trailing slash if present
  filename=${filename%/}
  filename=${filename%.hf}
  echo $filename

mkdir -p "$root_data_path/gemma_IF_3_160_$filename"

  # Run experiment
    CUDA_VISIBLE_DEVICES=0,1,2,3 python mola_training_gemma.py \
  --base_model "google/gemma-7b" \
  --data_path "$data_path" \
  --output_dir "$root_data_path/gemma_IF_3_160_$filename" \
  --batch_size 128 \
  --micro_batch_size 8 \
  --num_epochs 3 \
  --learning_rate 3e-4 \
  --cutoff_len 256 \
  --val_set_size 1 \
  --lora_r "8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8" \
  --lora_alpha 16 \
  --lora_dropout 0.05 \
  --lora_target_modules "q_proj,v_proj,k_proj,o_proj,gate_proj,down_proj,up_proj" \
  --number_experts "1,6,6,7,6,6,5,6,5,8,9,7,7,6,6,7,6,6,7,6,6,5,5,5,4,5,2,5" \
  --top_k "1,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2" \
  --train_on_inputs \
  --group_by_length \
  --add_eos_token \
  --wandb_run_name "gemma_IF_3_160_$filename"
done
