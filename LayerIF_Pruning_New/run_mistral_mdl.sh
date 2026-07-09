#!/bin/bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# cache_dir="$HOME/.cache/huggingface/hub/" # llm_weights/
cache_dir="llm_weights/" # local cache for testing

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
root_ratio_path="$PROJECT_ROOT/mdl/data"
model_name="Mistral-7B-v0.1"
model_path="mistralai/Mistral-7B-v0.1"

declare -a exp_dirs

# ratio_paths=(
#     "$root_ratio_path/rho_l_${model_name}_cola_mdl.json"
#     "$root_ratio_path/rho_l_${model_name}_mrpc_mdl.json"
#     "$root_ratio_path/rho_l_${model_name}_openbook_mdl.json"
#     "$root_ratio_path/rho_l_${model_name}_text_science_q_rebuttal_mdl.json"
#     "$root_ratio_path/rho_l_${model_name}_commonq_mdl.json"
# )

# ratio_paths=(
#     "$root_ratio_path/rho_l_${model_name}_cola_mdl_minmax_0.9.json"
#     "$root_ratio_path/rho_l_${model_name}_mrpc_mdl_minmax_0.9.json"
#     "$root_ratio_path/rho_l_${model_name}_openbook_mdl_minmax_0.9.json"
#     "$root_ratio_path/rho_l_${model_name}_text_science_q_rebuttal_mdl_minmax_0.9.json"
#     "$root_ratio_path/rho_l_${model_name}_commonq_mdl_minmax_0.9.json"
# )



DRY_RUN=false
# DRY_RUN=true --- IGNORE ---


ratio_family_first3_0_3=(
    "$root_ratio_path/rho_l_${model_name}_cola_mdl_minmax_0.55[0:3]_0_3.json"
    "$root_ratio_path/rho_l_${model_name}_mrpc_mdl_minmax_0.55[0:3]_0_3.json"
    # "$root_ratio_path/rho_l_${model_name}_openbook_mdl_minmax_0.55[0:3]_0_3.json"
    # "$root_ratio_path/rho_l_${model_name}_text_science_q_rebuttal_mdl_minmax_0.55[0:3]_0_3.json"
    "$root_ratio_path/rho_l_${model_name}_commonq_mdl_minmax_0.55[0:3]_0_3.json"
)

ratio_family_055_first1_0_4=(
    "$root_ratio_path/rho_l_${model_name}_cola_mdl_minmax_0.55[0:1]_0_4.json"
    "$root_ratio_path/rho_l_${model_name}_mrpc_mdl_minmax_0.55[0:1]_0_4.json"
    # "$root_ratio_path/rho_l_${model_name}_openbook_mdl_minmax_0.55[0:1]_0_4.json"
    # "$root_ratio_path/rho_l_${model_name}_text_science_q_rebuttal_mdl_minmax_0.55[0:1]_0_4.json"
    "$root_ratio_path/rho_l_${model_name}_commonq_mdl_minmax_0.55[0:1]_0_4.json"
)


ratio_family_055_first1_0_3=(
    "$root_ratio_path/rho_l_${model_name}_cola_mdl_minmax_0.55[0:1]_0_3.json"
    "$root_ratio_path/rho_l_${model_name}_mrpc_mdl_minmax_0.55[0:1]_0_3.json"
    # "$root_ratio_path/rho_l_${model_name}_openbook_mdl_minmax_0.55[0:1]_0_3.json"
    # "$root_ratio_path/rho_l_${model_name}_text_science_q_rebuttal_mdl_minmax_0.55[0:1]_0_3.json"
    "$root_ratio_path/rho_l_${model_name}_commonq_mdl_minmax_0.55[0:1]_0_3.json"
)

ratio_family_055_first1_0_4_next23_0_4=(
    "$root_ratio_path/rho_l_${model_name}_cola_mdl_minmax_0.55[0:1]_0_3[1:4]_0_4.json"
    "$root_ratio_path/rho_l_${model_name}_mrpc_mdl_minmax_0.55[0:1]_0_3[1:4]_0_4.json"
    # "$root_ratio_path/rho_l_${model_name}_openbook_mdl_minmax_0.55[0:1]_0_3[1:4]_0_4.json"
    # "$root_ratio_path/rho_l_${model_name}_text_science_q_rebuttal_mdl_minmax_0.55[0:1]_0_3[1:4]_0_4.json"
    "$root_ratio_path/rho_l_${model_name}_commonq_mdl_minmax_0.55[0:1]_0_3[1:4]_0_4.json"
)

# families=("ratio_family_055")
families=(
    "ratio_family_055_first1_0_4_next23_0_4"
    "ratio_family_first3_0_3"
    "ratio_family_055_first1_0_4"
    "ratio_family_055_first1_0_3"
)

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
        ratio_name=${ratio_name#rho_l_${model_name}_}

        # echo "$ratio_name"
        
        ratio_name_dir="results/${model_name}_MDL_${ratio_name}"
        mkdir -p -m 770 "$ratio_name_dir"
        # [-f $ratio_path] && cp -n "$ratio_path" "$ratio_name_dir/"
        cp -n "$ratio_path" "$ratio_name_dir/"

        exp_dirs+=("$ratio_name_dir")

        for method in "${prune_methods[@]}"; do
    
            save_dir="$ratio_name_dir/${method}"
            mkdir -p -m 770 "$save_dir"
            log_file="${save_dir}/log.txt"

            echo "---------------------------------------------"
            echo "Pruning with method: $method and $ratio_name ratio"
            echo "for model: ${model_path}"
            echo "Log file: ${log_file}"
            echo "---------------------------------------------"

            # 1. Build the base command using an array.
            # Note: We must prefix with 'env' so bash correctly passes the CUDA variable.
            cmd_args=(
                env CUDA_VISIBLE_DEVICES=2,3 CUDA_LAUNCH_BLOCKING=1 python main.py
                --model "$model_path"
                --cache_dir "$cache_dir"
                --prune_method "$method"
                --sparsity_ratio 0.5
                --save "$save_dir"
                --ww_metric "wanda"
                --ww_metric_cache "./data/${model_name}/"
                --epsilon 0.1
                --mdl_path "$ratio_path"
                --batch_size 8
                --track_norms
                --ratio_name "$ratio_name"
            )

            # 2. Dynamically inject the heavy evaluation flags if NOT a dry run.
            if [ "$DRY_RUN" = false ]; then
                # cmd_args+=(--eval_wikitext --eval_zero_shot)
                cmd_args+=(--eval_zero_shot)
            fi

            # 3. Execute the array with conditional logging.
            if [ "$DRY_RUN" = true ]; then
                # echo "  >>> DRY RUN: Outputting directly to terminal (eval disabled) <<<"
                
                log_file="${save_dir}/dry_run_log.txt"
                echo "  >>> DRY RUN: Output redirected to ${log_file} (eval disabled) <<<"
                "${cmd_args[@]}" > "$log_file" 2>&1
            else
                echo "  >>> FULL RUN: Output redirected to ${log_file} (eval enabled) <<<"
                "${cmd_args[@]}" > "$log_file" 2>&1
            fi

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
    # Both ratio_paths and the ${model_name}_MDL_* result directories are sorted
    # alphabetically by ratio name, so their order aligns correctly.
    # -------------------------------------------------------------------
    echo "============================================="
    echo "Generating norm plots for family: $current_family"
    echo "============================================="

    

    # Ensure the base plots directory exists before trying to write a log file there
    mkdir -p results/plots/
    plot_log="results/plots/plot_log_${current_family}.txt"
    echo "Plotting log will be saved to: $plot_log"

    # disabled 
    # --heat_map_repeat \
    # --ratio_overlay_repeat \

    python analysis/plot_norms.py \
        --experiment_dirs "${exp_dirs[@]}" \
        --output_dir results/plots/ \
        --ratio_paths "${ratio_paths[@]}" \
        > "$plot_log" 2>&1 #> 2>&1 | tee "$plot_log"
    echo "Finished $current_family. Moving to next..."
    
    
    sleep 5
done
