#!/bin/bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cache_dir="$HOME/.cache/huggingface/hub/"

declare -a exp_dirs

root_ratio_path="/data/mdl-layerIF/mdl/data"

ratio_family_055=(
    "$root_ratio_path/rho_l_gemma-7b_cola_mdl_minmax_0.55[0:1]_0_3[1:4]_0_4.json"
    "$root_ratio_path/rho_l_gemma-7b_mrpc_mdl_minmax_0.55[0:1]_0_3[1:4]_0_4.json"
    "$root_ratio_path/rho_l_gemma-7b_openbook_mdl_minmax_0.55[0:1]_0_3[1:4]_0_4.json"
    "$root_ratio_path/rho_l_gemma-7b_text_science_q_rebuttal_mdl_minmax_0.55[0:1]_0_3[1:4]_0_4.json"
    "$root_ratio_path/rho_l_gemma-7b_commonq_mdl_minmax_0.55[0:1]_0_3[1:4]_0_4.json"
)

families=("ratio_family_055")

prune_methods=("sparsegpt" "magnitude" "wanda")
# prune_methods=("sparsegpt")

for current_family in "${families[@]}"; do
    declare -n ratio_paths="$current_family"
    exp_dirs=() # reset exp_dirs for current family

    echo "============================================="
    echo "STARTING EXPERIMENT FAMILY: $current_family"
    echo "============================================="

    for ratio_path in "${ratio_paths[@]}"; do
        ratio_name=$(basename "$ratio_path" .json)
        ratio_name=${ratio_name#rho_l_gemma-7b_}

        ratio_name_dir="results/gemma_MDL_${ratio_name}"
        mkdir -p -m 770 "$ratio_name_dir"
        cp -n "$ratio_path" "$ratio_name_dir/"

        exp_dirs+=("$ratio_name_dir")

        for method in "${prune_methods[@]}"; do

            save_dir="$ratio_name_dir/${method}"
            mkdir -p -m 770 "$save_dir"
            log_file="${save_dir}/log.txt"

            echo "---------------------------------------------"
            echo "Pruning with method: $method and $ratio_name ratio"
            echo "for model: google/gemma-7b"
            echo "Log file: ${log_file}"
            echo "---------------------------------------------"
            # disabled --eval_wikitext \
            CUDA_VISIBLE_DEVICES=0,1 python main.py \
                --model google/gemma-7b \
                --cache_dir "$cache_dir" \
                --prune_method "$method" \
                --sparsity_ratio 0.5 \
                --save "$save_dir" \
                --ww_metric wanda \
                --ww_metric_cache ./data/gemma-7b/ \
                --epsilon 0.1 \
                --eval_zero_shot \
                --mdl_path "$ratio_path" \
                --batch_size 8 \
                --track_norms \
                --norm_n_samples 4 \
                --norm_seq_len 128 \
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

    # -------------------------------------------------------------------
    # After all experiments finish, generate the norm plots.
    # ratio_paths is passed explicitly so plot_norms.py does not have to
    # search the filesystem — the bash script is the authoritative source.
    # Both ratio_paths and the gemma_MDL_* result directories are sorted
    # alphabetically by ratio name, so their order aligns correctly.
    # -------------------------------------------------------------------
    echo "============================================="
    echo "Generating norm plots..."
    echo "============================================="
    python plot_norms.py \
        --experiment_dirs "${exp_dirs[@]}" \
        --output_dir plots/ \
        --ratio_paths "${ratio_paths[@]}"
    echo "Finished $current_family. Moving to next..."

    sleep 5

done