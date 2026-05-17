#!/bin/bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cache_dir="$HOME/.cache/huggingface/hub/" # llm_weights/

# root_ratio_path="/data/mdl-layerIF/mdl/data"
root_ratio_path="/data/mdl-layerIF/mdl/data_c"

# ratio_paths=(
#     "$root_ratio_path/rho_l_gemma-7b_cola_mdl.json"
#     "$root_ratio_path/rho_l_gemma-7b_mrpc_mdl.json"
#     "$root_ratio_path/rho_l_gemma-7b_openbook_mdl.json"
#     "$root_ratio_path/rho_l_gemma-7b_text_science_q_rebuttal_mdl.json"
#     "$root_ratio_path/rho_l_gemma-7b_commonq_mdl.json"
# )

# ratio_paths=(
#     "$root_ratio_path/rho_l_gemma-7b_cola_mdl_minmax_0.9.json"
#     "$root_ratio_path/rho_l_gemma-7b_mrpc_mdl_minmax_0.9.json"
#     "$root_ratio_path/rho_l_gemma-7b_openbook_mdl_minmax_0.9.json"
#     "$root_ratio_path/rho_l_gemma-7b_text_science_q_rebuttal_mdl_minmax_0.9.json"
#     "$root_ratio_path/rho_l_gemma-7b_commonq_mdl_minmax_0.9.json"
# )

# ratio_paths=(
#     "$root_ratio_path/rho_l_gemma-7b_cola_mdl_minmax_0.55[0:3]_0_3.json"
#     "$root_ratio_path/rho_l_gemma-7b_mrpc_mdl_minmax_0.55[0:3]_0_3.json"
#     "$root_ratio_path/rho_l_gemma-7b_openbook_mdl_minmax_0.55[0:3]_0_3.json"
#     "$root_ratio_path/rho_l_gemma-7b_text_science_q_rebuttal_mdl_minmax_0.55[0:3]_0_3.json"
#     "$root_ratio_path/rho_l_gemma-7b_commonq_mdl_minmax_0.55[0:3]_0_3.json"
# )

ratio_paths=(
    
    # for minmax 0.55, just run a subset of the ratios
    "$root_ratio_path/rho_l_gemma-7b_cola_mdl_minmax_0.55.json"
    "$root_ratio_path/rho_l_gemma-7b_mrpc_mdl_minmax_0.55.json"
    "$root_ratio_path/rho_l_gemma-7b_openbook_mdl_minmax_0.55.json"
    "$root_ratio_path/rho_l_gemma-7b_text_science_q_rebuttal_mdl_minmax_0.55.json"
    "$root_ratio_path/rho_l_gemma-7b_commonq_mdl_minmax_0.55.json"
    # for minmax 0.9, run all the ratios
    # "$root_ratio_path/rho_l_gemma-7b_cola_mdl_minmax_0.9.json"
    # "$root_ratio_path/rho_l_gemma-7b_mrpc_mdl_minmax_0.9.json"
    # "$root_ratio_path/rho_l_gemma-7b_openbook_mdl_minmax_0.9.json"
    # "$root_ratio_path/rho_l_gemma-7b_text_science_q_rebuttal_mdl_minmax_0.9.json"
    # "$root_ratio_path/rho_l_gemma-7b_commonq_mdl_minmax_0.9.json"

)

# prune_methods=("magnitude" "wanda" "sparsegpt")
prune_methods=("sparsegpt" "wanda")

for ratio_path in "${ratio_paths[@]}"; do
    ratio_name=$(basename "$ratio_path" .json)
    ratio_name=${ratio_name#rho_l_gemma-7b_}

    # echo "$ratio_name"
    
    ratio_name_dir="results_c/gemma_MDL_${ratio_name}"
    mkdir -p -m 770 "$ratio_name_dir"
    # [-f $ratio_path] && cp -n "$ratio_path" "$ratio_name_dir/"
    cp -n "$ratio_path" "$ratio_name_dir/"

    for method in "${prune_methods[@]}"; do
   
        save_dir="$ratio_name_dir/${method}"
        mkdir -p -m 770 "$save_dir"
        log_file="${save_dir}/log.txt"

        echo "---------------------------------------------"
        echo "Pruning with method: $method and $ratio_name ratio"
        echo "for model: google/gemma-7b"
        echo "Log file: ${log_file}"
        echo "---------------------------------------------"

        CUDA_VISIBLE_DEVICES=0,1,2 python main.py \
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

    sleep 10
done