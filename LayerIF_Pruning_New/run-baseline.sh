CUDA_VISIBLE_DEVICES=3 python  main.py \
    --model mistralai/Mistral-7B-v0.1 \
    --cache_dir llm_weights/ \
    --prune_method magnitude \
    --sparsity_ratio 0.5 \
    --save results/layerif/baseline/mistral-alpha-0.5-magnitude-alpha_peak \
    --ww_metric alpha_peak \
    --ww_metric_cache ./data/mistral-7b/ \
    --epsilon 0.1 \
    --eval_wikitext False \
    --eval_zero_shot

# CUDA_VISIBLE_DEVICES=0,1,2,3 python  main.py \
#     --model mistralai/Mistral-7B-v0.1 \
#     --cache_dir llm_weights/ \
#     --prune_method magnitude \
#     --sparsity_ratio 0.5 \
#     --save results/mistral-alpha-0.5-magnitude-mdl \
#     --ww_metric alpha_peak \
#     --ww_metric_cache ./data/mistral-7b/ \
#     --epsilon 0.1 \
#     --eval_wikitext False \
#     --eval_zero_shot \
#     --mdl


CUDA_VISIBLE_DEVICES=3 python  main.py \
    --model mistralai/Mistral-7B-v0.1 \
    --cache_dir llm_weights/ \
    --prune_method wanda \
    --sparsity_ratio 0.5 \
    --save results/layerif/baseline/mistral-alpha-0.5-wanda-alpha_peak  \
    --ww_metric alpha_peak \
    --ww_metric_cache ./data/mistral-7b/ \
    --epsilon 0.1 \
    --eval_wikitext False \
    --eval_zero_shot

# CUDA_VISIBLE_DEVICES=0,1,2,3 python  main.py \
#     --model mistralai/Mistral-7B-v0.1 \
#     --cache_dir llm_weights/ \
#     --prune_method wanda \
#     --sparsity_ratio 0.5 \
#     --save results/mistral-alpha-wanda-S0.5-mdl  \
#     --ww_metric alpha_peak \
#     --ww_metric_cache ./data/mistral-7b/ \
#     --epsilon 0.1 \
#     --eval_wikitext False \
#     --eval_zero_shot \
#     --mdl


CUDA_VISIBLE_DEVICES=3 python  main.py \
    --model mistralai/Mistral-7B-v0.1 \
    --cache_dir llm_weights/ \
    --prune_method sparsegpt \
    --sparsity_ratio 0.5 \
    --save results/layerif/baseline/mistral-alpha-0.5-sparsegpt-alpha_peak \
    --ww_metric alpha_peak \
    --ww_metric_cache ./data/mistral-7b/ \
    --epsilon 0.1 \
    --eval_wikitext False \
    --eval_zero_shot

