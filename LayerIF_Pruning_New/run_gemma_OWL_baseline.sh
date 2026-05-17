#!/bin/bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True # Enable PyTorch's expandable segments for better memory management during pruning and evaluation

prune_methods=("sparsegpt_OWL" "wanda_OWL" "magnitude_OWL")

cache_dir="$HOME/.cache/huggingface/hub/" # llm_weights/
experiment_name="OWL_baseline"
metric_name="OWL_prune_ratio_gemma_0.5"
model="google/gemma-7b"
model_name="gemma-7b"


for method in "${prune_methods[@]}"; do
    save_dir="results/${experiment_name}/gemma_${method}"
    mkdir -p -m 770 "$save_dir"
    log_file="${save_dir}/log.txt"

    echo "---------------------------------------------"
    echo "Pruning with method: $method with $metric_name metric at 50% sparsity"
    echo "for model: $model"
    echo "Log file: ${log_file}"
    echo "---------------------------------------------"

    CUDA_VISIBLE_DEVICES=0,1 python main.py \
        --model $model \
        --cache_dir "$cache_dir" \
        --prune_method "$method" \
        --sparsity_ratio 0.5 \
        --save ${save_dir} \
        --ww_metric ${metric_name} \
        --ww_metric_cache ./data/${model_name}/ \
        --epsilon 0.1 \
        --eval_zero_shot \
        --batch_size 8 \
        > "$log_file" 2>&1
done