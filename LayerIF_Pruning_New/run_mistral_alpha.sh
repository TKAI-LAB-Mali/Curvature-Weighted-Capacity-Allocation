CUDA_VISIBLE_DEVICES=4 python  main.py \
    --model mistralai/mistral-7B-v0.1 \
    --cache_dir llm_weights/ \
    --prune_method magnitude_ww \
    --sparsity_ratio 0.4 \
    --save results/mistral-alpha-0.4 \
    --ww_metric alpha_peak \
    --ww_metric_cache ./data/mistral-7b/ \
    --epsilon 0.1 \
    --eval_wikitext False \
    --eval_zero_shot


CUDA_VISIBLE_DEVICES=4 python  main.py \
    --model mistralai/mistral-7B-v0.1 \
    --cache_dir llm_weights/ \
    --prune_method wanda_ww \
    --sparsity_ratio 0.4 \
    --save results/mistral-alpha-0.4 \
    --ww_metric alpha_peak \
    --ww_metric_cache ./data/mistral-7b/ \
    --epsilon 0.1 \
    --eval_wikitext False \
    --eval_zero_shot




CUDA_VISIBLE_DEVICES=4 python  main.py \
    --model mistralai/mistral-7B-v0.1 \
    --cache_dir llm_weights/ \
    --prune_method sparsegpt_ww \
    --sparsity_ratio 0.4 \
    --save results/mistral-alpha-0.4 \
    --ww_metric alpha_peak \
    --ww_metric_cache ./data/mistral-7b/ \
    --epsilon 0.1 \
    --eval_wikitext False \
    --eval_zero_shot