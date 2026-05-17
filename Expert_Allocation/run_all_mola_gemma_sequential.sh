#!/bin/bash

root_data_path="/data/mdl-layerIF/Expert_Allocation"

# 1. Define Datasets (ScienceQ is now INCLUDED)
data_paths=(
  "$root_data_path/datasets/glue_mrpc_all.hf/"
  "$root_data_path/datasets/glue_cola_all.hf/"
  "$root_data_path/datasets/qa_text_scienceq_all.hf/"
  "$root_data_path/datasets/qa_commonq_all.hf/"
  "$root_data_path/datasets/qa_openbook_all.hf/"
)

# 2. Sequential Loop
for data_path in "${data_paths[@]}"; do
  filename=$(basename "$data_path")
  filename=${filename%/}
  filename=${filename%.hf}
  
  echo "=================================================="
  echo "STARTING SEQUENTIAL TRAINING FOR: $filename"
  echo "=================================================="

  

  output_dir="$root_data_path/training/mola3_5_7_8_baseline/gemma_IF_$filename"
  mkdir -p -m 770 "$output_dir"
  log_file="$output_dir/training.log"

  # Generate a random port to avoid address conflicts
  port=$((29500 + $RANDOM % 100))

  # 3. LAUNCH WITH TORCHRUN (All 4 GPUs work together)
  # --nproc_per_node=4 : Uses 4 GPUs
  # --micro_batch_size 1 : 1 per GPU * 4 GPUs = Effective Batch 4 (Prevents OOM)
  
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
  --number_experts "3,3,3,3,3,3,3,5,5,5,5,5,5,5,7,7,7,7,7,7,7,7,8,8,8,8,8,8" \
  --top_k "2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2" \
  --train_on_inputs \
  --group_by_length \
  --add_eos_token \
  --wandb_project "gemma_mola_3_5_7_8_run" \
  --wandb_run_name "gemma_mola_3_5_7_8_$filename" \
  > "$log_file" 2>&1

  # Check exit status
  if [ $? -eq 0 ]; then
    echo "SUCCESS: $filename finished."
  else
    echo "FAILURE: $filename crashed. Check $log_file"
  fi

  # 4. COOL DOWN PAUSE
  # Waiting 10 seconds helps ensure VRAM is fully cleared before the next run starts
  sleep 10
  
done

echo "All sequential jobs finished."