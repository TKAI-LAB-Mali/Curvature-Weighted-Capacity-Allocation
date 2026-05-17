#!/bin/bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cache_dir="$HOME/.cache/huggingface/hub/" # llm_weights/

root_ratio_path="/data/mdl-layerIF/mdl/data_dp"

declare -a exp_dirs

DRY_RUN=true

ratio_family_055_tau1=(
    # subset 0 with minmax 0.55
    "$root_ratio_path/rho_l_gemma-7b_cola_mdl_minmax_0.55.json"
    "$root_ratio_path/rho_l_gemma-7b_mrpc_mdl_minmax_0.55.json"
    "$root_ratio_path/rho_l_gemma-7b_openbook_mdl_minmax_0.55.json"
    "$root_ratio_path/rho_l_gemma-7b_text_science_q_rebuttal_mdl_minmax_0.55.json"
    "$root_ratio_path/rho_l_gemma-7b_commonq_mdl_minmax_0.55.json")
     # subset 1 with minmax 0.55[0:4]_0_3
ratio_family_055_tau1_first4_0_3=(
    "$root_ratio_path/rho_l_gemma-7b_cola_mdl_minmax_0.55[0:4]_0_3.json"
    "$root_ratio_path/rho_l_gemma-7b_mrpc_mdl_minmax_0.55[0:4]_0_3.json"
    "$root_ratio_path/rho_l_gemma-7b_openbook_mdl_minmax_0.55[0:4]_0_3.json"
    "$root_ratio_path/rho_l_gemma-7b_text_science_q_rebuttal_mdl_minmax_0.55[0:4]_0_3.json"
    "$root_ratio_path/rho_l_gemma-7b_commonq_mdl_minmax_0.55[0:4]_0_3.json")
    # subset 2 with minmax 0.9
ratio_family_09_tau1=(
    "$root_ratio_path/rho_l_gemma-7b_cola_mdl_minmax_0.9.json"
    "$root_ratio_path/rho_l_gemma-7b_mrpc_mdl_minmax_0.9.json"
    "$root_ratio_path/rho_l_gemma-7b_openbook_mdl_minmax_0.9.json"
    "$root_ratio_path/rho_l_gemma-7b_text_science_q_rebuttal_mdl_minmax_0.9.json"
    "$root_ratio_path/rho_l_gemma-7b_commonq_mdl_minmax_0.9.json")
    # subset 3 with minmax 0.9[0:4]_0_3
ratio_family_09_tau1_first4_0_3=(
    "$root_ratio_path/rho_l_gemma-7b_cola_mdl_minmax_0.9[0:4]_0_3.json"
    "$root_ratio_path/rho_l_gemma-7b_mrpc_mdl_minmax_0.9[0:4]_0_3.json"
    "$root_ratio_path/rho_l_gemma-7b_openbook_mdl_minmax_0.9[0:4]_0_3.json"
    "$root_ratio_path/rho_l_gemma-7b_text_science_q_rebuttal_mdl_minmax_0.9[0:4]_0_3.json"
    "$root_ratio_path/rho_l_gemma-7b_commonq_mdl_minmax_0.9[0:4]_0_3.json"    
)

ratio_family_09_tau_4=(
    # Subset 0 with minmax 0.9 and tau -4
    "$root_ratio_path/rho_l_gemma-7b_cola_mdl_tau_-4_mean_minmax_0.9.json"
    "$root_ratio_path/rho_l_gemma-7b_mrpc_mdl_tau_-4_mean_minmax_0.9.json"
    # "$root_ratio_path/rho_l_gemma-7b_openbook_mdl_tau_-4_mean_minmax_0.9.json"
    # "$root_ratio_path/rho_l_gemma-7b_text_science_q_rebuttal_mdl_tau_-4_mean_minmax_0.9.json"
    "$root_ratio_path/rho_l_gemma-7b_commonq_mdl_tau_-4_mean_minmax_0.9.json")
    # Subset 1 with minmax 0.55 and tau -4
ratio_family_055_tau_4=("$root_ratio_path/rho_l_gemma-7b_cola_mdl_tau_-4_mean_minmax_0.55.json"
    "$root_ratio_path/rho_l_gemma-7b_mrpc_mdl_tau_-4_mean_minmax_0.55.json"
    # "$root_ratio_path/rho_l_gemma-7b_openbook_mdl_tau_-4_mean_minmax_0.55.json"
    # "$root_ratio_path/rho_l_gemma-7b_text_science_q_rebuttal_mdl_tau_-4_mean_minmax_0.55.json"
    "$root_ratio_path/rho_l_gemma-7b_commonq_mdl_tau_-4_mean_minmax_0.55.json") 
    # Subset 0 with minmax 0.9 and tau 4
ratio_family_09_tau4=("$root_ratio_path/rho_l_gemma-7b_cola_mdl_tau_4_mean_minmax_0.9.json"
    "$root_ratio_path/rho_l_gemma-7b_mrpc_mdl_tau_4_mean_minmax_0.9.json"
    # "$root_ratio_path/rho_l_gemma-7b_openbook_mdl_tau_4_mean_minmax_0.9.json"
    # "$root_ratio_path/rho_l_gemma-7b_text_science_q_rebuttal_mdl_tau_4_mean_minmax_0.9.json"
    "$root_ratio_path/rho_l_gemma-7b_commonq_mdl_tau_4_mean_minmax_0.9.json")
    # Subset 1 with minmax 0.55 and tau 4

ratio_family_055_tau4=("$root_ratio_path/rho_l_gemma-7b_cola_mdl_tau_4_mean_minmax_0.55.json"
    "$root_ratio_path/rho_l_gemma-7b_mrpc_mdl_tau_4_mean_minmax_0.55.json"
    # "$root_ratio_path/rho_l_gemma-7b_openbook_mdl_tau_4_mean_minmax_0.55.json"
    # "$root_ratio_path/rho_l_gemma-7b_text_science_q_rebuttal_mdl_tau_4_mean_minmax_0.55.json"
    "$root_ratio_path/rho_l_gemma-7b_commonq_mdl_tau_4_mean_minmax_0.55.json" 
)

families=("ratio_family_09_tau_4" "ratio_family_055_tau_4" 
            "ratio_family_09_tau4" "ratio_family_055_tau4"
            "ratio_family_09_tau1" "ratio_family_09_tau1_first4_0_3"
            "ratio_family_055_tau1" "ratio_family_055_tau1_first4_0_3")

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
        # ratio_name=${ratio_name%_minmax_0.9}

        # echo "$ratio_name"
        
        
        ratio_name_dir="depth_prior_results/gemma_MDL_${ratio_name}"
        mkdir -p -m 770 "$ratio_name_dir"
        # [-f $ratio_path] && cp -n "$ratio_path" "$ratio_name_dir/"
        cp -n "$ratio_path" "$ratio_name_dir/"

        exp_dirs+=("$ratio_name_dir")


        for method in "${prune_methods[@]}"; do
    
            save_dir="${ratio_name_dir}/${method}"
            mkdir -p -m 770 "$save_dir"
            log_file="${save_dir}/log.txt"

            echo "---------------------------------------------"
            echo "Pruning with method: $method and $ratio_name ratio"
            echo "for model: google/gemma-7b"
            echo "Log file: ${log_file}"
            echo "---------------------------------------------"

            # 1. Build the base command using an array.
            # Note: We must prefix with 'env' so bash correctly passes the CUDA variable.
            cmd_args=(
                env CUDA_VISIBLE_DEVICES=2,3 python main.py
                --model "google/gemma-7b"
                --cache_dir "$cache_dir"
                --prune_method "$method"
                --sparsity_ratio 0.5
                --save "$save_dir"
                --ww_metric "wanda"
                --ww_metric_cache "./data/gemma-7b/"
                --epsilon 0.1
                --mdl_path "$ratio_path"
                --batch_size 8
                --track_norms
                --ratio_name "$ratio_name"
            )

            # 2. Dynamically inject the heavy evaluation flags if NOT a dry run.
            if [ "$DRY_RUN" = false ]; then
                cmd_args+=(--eval_wikitext --eval_zero_shot)
            fi

            # 3. Execute the array with conditional logging.
            if [ "$DRY_RUN" = true ]; then
                
                
                log_file="${save_dir}/dry_run_log.txt"
                echo "  >>> DRY RUN: Outputting directly to ${log_file} (eval disabled) <<<"

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
    # Both ratio_paths and the gemma_MDL_* result directories are sorted
    # alphabetically by ratio name, so their order aligns correctly.
    # -------------------------------------------------------------------
    echo "============================================="
    echo "Generating norm plots for family: $current_family"
    echo "============================================="

    

    # Ensure the base plots directory exists before trying to write a log file there
    mkdir -p depth_prior_results/plots/
    plot_log="depth_prior_results/plots/plot_log_${current_family}.txt"

    echo "Plotting log will be saved to: $plot_log"

    # disabled 
    # --heat_map_repeat \
    # --ratio_overlay_repeat \

    python analysis/plot_norms.py \
        --experiment_dirs "${exp_dirs[@]}" \
        --output_dir depth_prior_results/plots/ \
        --ratio_paths "${ratio_paths[@]}" \
        > "$plot_log" 2>&1 #> 2>&1 | tee "$plot_log"
    echo "Finished $current_family. Moving to next..."

    sleep 5
done