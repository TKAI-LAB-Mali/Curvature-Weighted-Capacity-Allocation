# CUDA_VISIBLE_DEVICES=0,1,2,3 python  main.py \
#     --model google/gemma-7b \
#     --cache_dir llm_weights/ \
#     --prune_method magnitude \
#     --sparsity_ratio 0.7 \
#     --save results/gemma-alpha \
#     --ww_metric alpha_peak \
#     --ww_metric_cache ./data/gemma-7b/ \
#     --epsilon 0.1 \
#     --eval_wikitext False \
#     --eval_zero_shot


# CUDA_VISIBLE_DEVICES=0,1,2,3 python  main.py \
#     --model google/gemma-7b \
#     --cache_dir llm_weights/ \
#     --prune_method wanda \
#     --sparsity_ratio 0.7 \
#     --save results/gemma-alpha \
#     --ww_metric alpha_peak \
#     --ww_metric_cache ./data/gemma-7b/ \
#     --epsilon 0.1 \
#     --eval_wikitext False \
#     --eval_zero_shot




CUDA_VISIBLE_DEVICES=0,1,2,3 python  main.py \
    --model google/gemma-7b \
    --cache_dir llm_weights/ \
    --prune_method sparsegpt_ww \
    --sparsity_ratio 0.7 \
    --save results/gemma-alpha \
    --ww_metric alpha_peak \
    --ww_metric_cache ./data/gemma-7b/ \
    --epsilon 0.1 \
    --eval_wikitext False \
    --eval_zero_shot