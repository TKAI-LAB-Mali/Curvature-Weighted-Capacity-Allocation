#!/bin/bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cache_dir="$HOME/.cache/huggingface/hub/" # llm_weights/

root_ratio_path="/data/mdl-layerIF/mdl/data/depth_penalty"

# 1. We define the exact JSON files that have missing results
missing_ratios=(
    # minmax 0.9 linear missing sparsegpt
    "rho_l_gemma-7b_cola_mdl_linear_strength_10.0_p_10.0_minmax_0.9[0:3]_0_3.json"
    "rho_l_gemma-7b_mrpc_mdl_linear_strength_10.0_p_10.0_minmax_0.9[0:3]_0_3.json"
    "rho_l_gemma-7b_openbook_mdl_linear_strength_10.0_p_10.0_minmax_0.9[0:3]_0_3.json"
    "rho_l_gemma-7b_text_science_q_rebuttal_mdl_linear_strength_10.0_p_10.0_minmax_0.9[0:3]_0_3.json"
    
    # minmax 0.9 u_curve missing sparsegpt
    "rho_l_gemma-7b_text_science_q_rebuttal_mdl_u_curve_strength_10.0_p_10.0_minmax_0.9[0:3]_0_3.json"
    
    # minmax 0.55 linear missing sparsegpt
    "rho_l_gemma-7b_cola_mdl_linear_strength_10.0_p_10.0_minmax_0.55[0:3]_0_3.json"
    
    # minmax 0.55 u_curve missing combinations
    "rho_l_gemma-7b_mrpc_mdl_u_curve_strength_10.0_p_10.0_minmax_0.55[0:3]_0_3.json"
    "rho_l_gemma-7b_commonq_mdl_u_curve_strength_10.0_p_10.0_minmax_0.55[0:3]_0_3.json"
    "rho_l_gemma-7b_commonq_mdl_u_curve_strength_10.0_p_10.0_minmax_0.55[0:3]_0_3.json"
    "rho_l_gemma-7b_openbook_mdl_u_curve_strength_10.0_p_10.0_minmax_0.55[0:3]_0_3.json"
    "rho_l_gemma-7b_text_science_q_rebuttal_mdl_u_curve_strength_10.0_p_10.0_minmax_0.55[0:3]_0_3.json"
    "rho_l_gemma-7b_text_science_q_rebuttal_mdl_u_curve_strength_10.0_p_10.0_minmax_0.55[0:3]_0_3.json"
)

# 2. We define the exact missing prune methods corresponding to the files above
missing_methods=(
    "sparsegpt"   # cola linear 0.9
    "sparsegpt"   # mrpc linear 0.9
    "sparsegpt"   # openbook linear 0.9
    "sparsegpt"   # scienceQ linear 0.9
    "sparsegpt"   # scienceQ u_curve 0.9
    "sparsegpt"   # cola linear 0.55
    "sparsegpt"   # mrpc u_curve 0.55
    "wanda"       # commonq u_curve 0.55
    "sparsegpt"   # commonq u_curve 0.55
    "sparsegpt"   # openbook u_curve 0.55
    "wanda"       # scienceQ u_curve 0.55
    "sparsegpt"   # scienceQ u_curve 0.55
)

# 3. Iterate through both arrays simultaneously
for i in "${!missing_ratios[@]}"; do
    ratio_file="${missing_ratios[$i]}"
    method="${missing_methods[$i]}"
    ratio_path="$root_ratio_path/$ratio_file"

    # Extract clean name
    ratio_name=$(basename "$ratio_path" .json)
    ratio_name=${ratio_name#rho_l_gemma-7b_}
    
    # Directory setup
    ratio_name_dir="depth_penalty_results/gemma_MDL_${ratio_name}"
    mkdir -p -m 770 "$ratio_name_dir"
    cp -n "$ratio_path" "$ratio_name_dir/"

    save_dir="${ratio_name_dir}/${method}"
    mkdir -p -m 770 "$save_dir"
    log_file="${save_dir}/log.txt"

    echo "---------------------------------------------"
    echo "Experiment $((i+1)) / ${#missing_ratios[@]}"
    echo "Pruning with method: $method and $ratio_name ratio"
    echo "for model: google/gemma-7b"
    echo "Log file: ${log_file}"
    echo "---------------------------------------------"

    CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py \
        --model google/gemma-7b \
        --cache_dir "$cache_dir" \
        --prune_method "$method" \
        --sparsity_ratio 0.5 \
        --save "$save_dir" \
        --ww_metric wanda \
        --ww_metric_cache ./data/gemma-7b/ \
        --epsilon 0.1 \
        --eval_wikitext False \
        --eval_zero_shot \
        --mdl_path "$ratio_path" \
        --batch_size 8 \
        > "$log_file" 2>&1

    if [ $? -eq 0 ]; then 
        echo "SUCCESS: Evaluated on Pruning with method: $method and $ratio_name ratio"
    else
        echo "FAILURE: Failed to prune with $method and $ratio_name ratio"
    fi

    sleep 10
done