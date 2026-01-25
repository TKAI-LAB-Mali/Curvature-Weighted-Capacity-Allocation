#!/bin/bash

root_data_path="/data/mdl-layerIF/Expert_Allocation"

data_paths=(
  "$root_data_path/datasets/glue_mrpc_all.hf/"
  "$root_data_path/datasets/glue_cola_all.hf/"
  "$root_data_path/datasets/qa_text_scienceq_all.hf/"
  "$root_data_path/datasets/qa_commonq_all.hf/"
  "$root_data_path/datasets/qa_openbook_all.hf/"
)

# Counter to track which GPU to use (0, 1, 2, or 3)
gpu_id=0

for data_path in "${data_paths[@]}"; do
  filename=$(basename "$data_path")
  filename=${filename%/}
  filename=${filename%.hf}
  
  # --- LOGIC TO SELECT EXPERTS ---
  case "$filename" in
    *"commonq"*)
      current_experts="1,5,6,5,4,5,5,6,7,7,7,6,5,5,7,8,7,6,8,6,7,7,6,5,5,5,4,5"
      current_top_k="1,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2"
      ;;
    *"mrpc"*)
      current_experts="1,6,6,6,6,6,5,6,5,7,8,7,7,6,6,7,6,6,7,7,6,6,5,5,4,5,3,5"
      current_top_k="1,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2"
      ;;
    *"cola"*)
      current_experts="1,4,7,7,8,7,8,8,8,7,7,7,7,5,6,7,6,7,5,5,5,5,5,5,4,3,3,3"
      current_top_k="1,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2"
      ;;
    *"openbook"*)
      current_experts="1,5,7,6,5,6,5,8,8,8,7,7,5,4,5,7,6,4,7,7,6,7,7,3,4,6,4,5"
      current_top_k="1,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2"
      ;;
    *"scienceq"*)
      current_experts="1,12,7,7,5,5,4,5,6,8,11,5,4,4,3,7,3,3,6,6,6,6,6,7,5,5,3,10" 
      current_top_k="1,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2"
      ;;
    *)
      echo "Using default config for $filename"
      current_experts="1,5,7,6,5,6,5,8,8,8,7,7,5,4,5,7,6,4,7,7,6,7,7,3,4,6,4,5"
      current_top_k="1,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2"
      ;;
  esac
  output_dir="$root_data_path/training/gemma_IF_$filename"
  mkdir -p -m 770 "$output_dir"
  log_file="$output_dir/training.log"

  echo "Launching $filename on GPU $gpu_id..."

  # 3. PARALLEL EXECUTION MAGIC
  # We use the variable $gpu_id to assign ONE GPU per task.
  # We add '&' at the end to run it in the background immediately.
  
  CUDA_VISIBLE_DEVICES=$gpu_id python -u mola_training_gemma.py \
  --base_model "google/gemma-7b" \
  --data_path "$data_path" \
  --output_dir "$output_dir" \
  --batch_size 128 \
  --micro_batch_size 4 \
  --num_epochs 10 \
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
  --wandb_run_name "gemma_IF_$filename" \
  > "$log_file" 2>&1 &  

  # 4. SAFETY PAUSE
  # Wait 60 seconds before launching the next one.
  # This lets the first job load the model into VRAM and settle down.
  # Without this, launching 5 jobs instantly might crash your System RAM (CPU)
  sleep 60
  
  # Cycle the GPU ID for the next loop (0 -> 1 -> 2 -> 3 -> 0)
  gpu_id=$(( (gpu_id + 1) % 4 ))
  
done

echo "All jobs launched! Monitor progress using 'nvidia-smi'."
wait # Wait for all background jobs to finish before the script exits