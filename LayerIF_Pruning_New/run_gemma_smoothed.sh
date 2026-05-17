#!/bin/bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True # Enable PyTorch's expandable segments for better memory management during pruning and evaluation

prune_methods=("magnitude_ww" "wanda_ww" "sparsegpt_ww")
# prune_methods=("sparsegpt_ww") # For quick testing, only run sparsegpt_ww. Remove this line to run all methods.
# prune_methods=("sparsegpt_ww")
cache_dir="$HOME/.cache/huggingface/hub/" # llm_weights/


for method in "${prune_methods[@]}"; do
    save_dir="results/repeat/gemma-IF-smoothed_0.5_layerIF_${method}"
    mkdir -p -m 770 "$save_dir"
    log_file="${save_dir}/log.txt"

    echo "---------------------------------------------"
    echo "Pruning with method: $method with IF-300-96-smoothed metric at 50% sparsity"
    echo "for model: google/gemma-7b"
    echo "Log file: ${log_file}"
    echo "---------------------------------------------"

    CUDA_VISIBLE_DEVICES=0,1,2 python main.py \
        --model google/gemma-7b \
        --cache_dir "$cache_dir" \
        --prune_method "$method" \
        --sparsity_ratio 0.5 \
        --save ${save_dir} \
        --ww_metric IF-300-96-smoothed \
        --ww_metric_cache ./data/gemma-7b/ \
        --epsilon 0.1 \
        --eval_wikitext False \
        --eval_zero_shot \
        --batch_size 8 \
        > "$log_file" 2>&1
done