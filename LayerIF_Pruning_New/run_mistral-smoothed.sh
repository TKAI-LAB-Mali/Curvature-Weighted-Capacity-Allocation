CUDA_VISIBLE_DEVICES=5 python  main.py \
    --model mistralai/Mistral-7B-v0.1 \
    --cache_dir llm_weights/ \
    --prune_method magnitude_ww \
    --sparsity_ratio 0.4 \
    --save results/mistral-IF-smoothed-0.4 \
    --ww_metric IF-300-96-smoothed \
    --ww_metric_cache ./data/mistral-7b/ \
    --epsilon 0.3 \
    --eval_wikitext False \
    --eval_zero_shot

CUDA_VISIBLE_DEVICES=5 python  main.py \
    --model mistralai/Mistral-7B-v0.1 \
    --cache_dir llm_weights/ \
    --prune_method wanda_ww \
    --sparsity_ratio 0.4 \
    --save results/mistral-IF-smoothed-0.4 \
    --ww_metric IF-300-96-smoothed \
    --ww_metric_cache ./data/mistral-7b/ \
    --epsilon 0.1 \
    --eval_wikitext False \
    --eval_zero_shot

CUDA_VISIBLE_DEVICES=5 python  main.py \
    --model mistralai/Mistral-7B-v0.1 \
    --cache_dir llm_weights/ \
    --prune_method sparsegpt_ww \
    --sparsity_ratio 0.4 \
    --save results/mistral-IF-smoothed-0.4 \
    --ww_metric IF-300-96-smoothed \
    --ww_metric_cache ./data/mistral-7b/ \
    --epsilon 0.1 \
    --eval_wikitext False \
    --eval_zero_shot

# CUDA_VISIBLE_DEVICES=6 python  main.py \
#     --model mistralai/Mistral-7B-v0.1 \
#     --cache_dir llm_weights/ \
#     --prune_method magnitude_ww \
#     --sparsity_ratio 0.5 \
#     --save results/mistral-IF-smoothed-0.5 \
#     --ww_metric IF-300-96-smoothed \
#     --ww_metric_cache ./data/mistral-7b/ \
#     --epsilon 0.4 \
#     --eval_wikitext False \
#     --eval_zero_shot


# CUDA_VISIBLE_DEVICES=6 python  main.py \
#     --model mistralai/Mistral-7B-v0.1 \
#     --cache_dir llm_weights/ \
#     --prune_method magnitude_ww \
#     --sparsity_ratio 0.5 \
#     --save results/mistral-IF-smoothed-0.5 \
#     --ww_metric IF-300-96-smoothed \
#     --ww_metric_cache ./data/mistral-7b/ \
#     --epsilon 0.5 \
#     --eval_wikitext False \
#     --eval_zero_shot



# CUDA_VISIBLE_DEVICES=6 python  main.py \
#     --model mistralai/Mistral-7B-v0.1 \
#     --cache_dir llm_weights/ \
#     --prune_method magnitude_ww \
#     --sparsity_ratio 0.5 \
#     --save results/mistral-IF-smoothed-0.5 \
#     --ww_metric IF-300-96-smoothed \
#     --ww_metric_cache ./data/mistral-7b/ \
#     --epsilon 0.5 \
#     --eval_wikitext False \
#     --eval_zero_shot
#         # --mapping_type "linear" \

# CUDA_VISIBLE_DEVICES=6 python  main.py \
#     --model mistralai/Mistral-7B-v0.1 \
#     --cache_dir llm_weights/ \
#     --prune_method wanda_ww \
#     --sparsity_ratio 0.5 \
#     --save results/mistral-IF-smoothed-0.5 \
#     --ww_metric IF-300-96-smoothed \
#     --ww_metric_cache ./data/mistral-7b/ \
#     --epsilon 0.1 \
#     --eval_wikitext False \
#     --eval_zero_shot
#         # --mapping_type "linear" \

# CUDA_VISIBLE_DEVICES=6 python  main.py \
#     --model mistralai/Mistral-7B-v0.1 \
#     --cache_dir llm_weights/ \
#     --prune_method wanda_ww \
#     --sparsity_ratio 0.5 \
#     --save results/mistral-IF-smoothed-0.5 \
#     --ww_metric IF-300-96-smoothed \
#     --ww_metric_cache ./data/mistral-7b/ \
#     --epsilon 0.2 \
#     --eval_wikitext False \
#     --eval_zero_shot
#         # --mapping_type "linear" \

# CUDA_VISIBLE_DEVICES=6 python  main.py \
#     --model mistralai/Mistral-7B-v0.1 \
#     --cache_dir llm_weights/ \
#     --prune_method wanda_ww \
#     --sparsity_ratio 0.5 \
#     --save results/mistral-IF-smoothed-0.5 \
#     --ww_metric IF-300-96-smoothed \
#     --ww_metric_cache ./data/mistral-7b/ \
#     --epsilon 0.3 \
#     --eval_wikitext False \
#     --eval_zero_shot
#         # --mapping_type "linear" \

# CUDA_VISIBLE_DEVICES=6 python  main.py \
#     --model mistralai/Mistral-7B-v0.1 \
#     --cache_dir llm_weights/ \
#     --prune_method wanda_ww \
#     --sparsity_ratio 0.5 \
#     --save results/mistral-IF-smoothed-0.5 \
#     --ww_metric IF-300-96-smoothed \
#     --ww_metric_cache ./data/mistral-7b/ \
#     --epsilon 0.4 \
#     --eval_wikitext False \
#     --eval_zero_shot
#         # --mapping_type "linear" \


# CUDA_VISIBLE_DEVICES=6 python  main.py \
#     --model mistralai/Mistral-7B-v0.1 \
#     --cache_dir llm_weights/ \
#     --prune_method wanda_ww \
#     --sparsity_ratio 0.5 \
#     --save results/mistral-IF-smoothed-0.5 \
#     --ww_metric IF-300-96-smoothed \
#     --ww_metric_cache ./data/mistral-7b/ \
#     --epsilon 0.5 \
#     --eval_wikitext False \
#     --eval_zero_shot
#         # --mapping_type "linear" \


# CUDA_VISIBLE_DEVICES=6 python  main.py \
#     --model mistralai/Mistral-7B-v0.1 \
#     --cache_dir llm_weights/ \
#     --prune_method sparsegpt_ww \
#     --sparsity_ratio 0.5 \
#     --save results/mistral-IF-smoothed-0.5 \
#     --ww_metric IF-300-96-smoothed \
#     --ww_metric_cache ./data/mistral-7b/ \
#     --epsilon 0.1 \
#     --eval_wikitext False \
#     --eval_zero_shot
#         # --mapping_type "linear" \

# CUDA_VISIBLE_DEVICES=6 python  main.py \
#     --model mistralai/Mistral-7B-v0.1 \
#     --cache_dir llm_weights/ \
#     --prune_method sparsegpt_ww \
#     --sparsity_ratio 0.5 \
#     --save results/mistral-IF-smoothed-0.5 \
#     --ww_metric IF-300-96-smoothed \
#     --ww_metric_cache ./data/mistral-7b/ \
#     --epsilon 0.2 \
#     --eval_wikitext False \
#     --eval_zero_shot
#         # --mapping_type "linear" \

# CUDA_VISIBLE_DEVICES=6 python  main.py \
#     --model mistralai/Mistral-7B-v0.1 \
#     --cache_dir llm_weights/ \
#     --prune_method sparsegpt_ww \
#     --sparsity_ratio 0.5 \
#     --save results/mistral-IF-smoothed-0.5 \
#     --ww_metric IF-300-96-smoothed \
#     --ww_metric_cache ./data/mistral-7b/ \
#     --epsilon 0.3 \
#     --eval_wikitext False \
#     --eval_zero_shot
#         # --mapping_type "linear" \

# CUDA_VISIBLE_DEVICES=6 python  main.py \
#     --model mistralai/Mistral-7B-v0.1 \
#     --cache_dir llm_weights/ \
#     --prune_method sparsegpt_ww \
#     --sparsity_ratio 0.5 \
#     --save results/mistral-IF-smoothed-0.5 \
#     --ww_metric IF-300-96-smoothed \
#     --ww_metric_cache ./data/mistral-7b/ \
#     --epsilon 0.4 \
#     --eval_wikitext False \
#     --eval_zero_shot
#         # --mapping_type "linear" \

# CUDA_VISIBLE_DEVICES=6 python  main.py \
#     --model mistralai/Mistral-7B-v0.1 \
#     --cache_dir llm_weights/ \
#     --prune_method sparsegpt_ww \
#     --sparsity_ratio 0.5 \
#     --save results/mistral-IF-smoothed-0.5 \
#     --ww_metric IF-300-96-smoothed \
#     --ww_metric_cache ./data/mistral-7b/ \
#     --epsilon 0.5 \
#     --eval_wikitext False \
#     --eval_zero_shot
#         # --mapping_type "linear" \