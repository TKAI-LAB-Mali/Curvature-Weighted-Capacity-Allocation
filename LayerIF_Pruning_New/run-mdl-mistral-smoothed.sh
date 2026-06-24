CUDA_VISIBLE_DEVICES=2 python  main.py \
    --model mistralai/Mistral-7B-v0.1 \
    --cache_dir llm_weights/ \
    --prune_method sparsegpt_ww \
    --sparsity_ratio 0.5 \
    --save results/mdl/sparsegpt/mistral-IF-smoothed-0.5-min-0.51-scienceq \
    --ww_metric IF-300-96-smoothed \
    --ww_metric_cache ./data/mistral-7b/ \
    --epsilon 0.1 \
    --eval_wikitext False \
    --eval_zero_shot \
    --mdl text_science_q_rebuttal

CUDA_VISIBLE_DEVICES=1 python  main.py \
    --model mistralai/Mistral-7B-v0.1 \
    --cache_dir llm_weights/ \
    --prune_method magnitude_ww \
    --sparsity_ratio 0.5 \
    --save results/mdl/magnitude_ww/mistral-IF-smoothed-0.5-min-0.51-commonq \
    --ww_metric IF-300-96-smoothed \
    --ww_metric_cache ./data/mistral-7b/ \
    --epsilon 0.1 \
    --eval_wikitext False \
    --eval_zero_shot \
    --mdl commonq

CUDA_VISIBLE_DEVICES=2 python  main.py \
    --model mistralai/Mistral-7B-v0.1 \
    --cache_dir llm_weights/ \
    --prune_method magnitude_ww \
    --sparsity_ratio 0.5 \
    --save results/mdl/magnitude_ww/mistral-IF-smoothed-0.5-min-0.51-openbook \
    --ww_metric IF-300-96-smoothed \
    --ww_metric_cache ./data/mistral-7b/ \
    --epsilon 0.1 \
    --eval_wikitext False \
    --eval_zero_shot \
    --mdl openbook
    
CUDA_VISIBLE_DEVICES=3 python  main.py \
    --model mistralai/Mistral-7B-v0.1 \
    --cache_dir llm_weights/ \
    --prune_method magnitude_ww \
    --sparsity_ratio 0.5 \
    --save results/mdl/magnitude_ww/mistral-IF-smoothed-0.5-min-0.51-scienceq \
    --ww_metric IF-300-96-smoothed \
    --ww_metric_cache ./data/mistral-7b/ \
    --epsilon 0.1 \
    --eval_wikitext False \
    --eval_zero_shot \
    --mdl text_science_q_rebuttal