CUDA_VISIBLE_DEVICES=2 python  main.py \
    --model mistralai/Mistral-7B-v0.1 \
    --cache_dir llm_weights/ \
    --prune_method magnitude_OWL \
    --sparsity_ratio 0.5 \
    --save results/layerif/OWL/mistral-OWL-0.5-magnitude \
    --ww_metric OWL_prune_ratio_mistral_0.5 \
    --ww_metric_cache ./data/mistral-7b/ \
    --epsilon 0.1 \
    --eval_wikitext False \
    --eval_zero_shot



CUDA_VISIBLE_DEVICES=2 python  main.py \
    --model mistralai/Mistral-7B-v0.1 \
    --cache_dir llm_weights/ \
    --prune_method wanda_OWL \
    --sparsity_ratio 0.5 \
    --save results/layerif/OWL/mistral-OWL-0.5-wanda \
    --ww_metric OWL_prune_ratio_mistral_0.5 \
    --ww_metric_cache ./data/mistral-7b/ \
    --epsilon 0.1 \
    --eval_wikitext False \
    --eval_zero_shot


CUDA_VISIBLE_DEVICES=1 python  main.py \
    --model mistralai/Mistral-7B-v0.1 \
    --cache_dir llm_weights/ \
    --prune_method sparsegpt_OWL \
    --sparsity_ratio 0.5 \
    --save results/layerif/OWL/mistral-OWL-0.5-sparsegpt \
    --ww_metric OWL_prune_ratio_mistral_0.5 \
    --ww_metric_cache ./data/mistral-7b/ \
    --epsilon 0.1 \
    --eval_wikitext False \
    --eval_zero_shot



CUDA_VISIBLE_DEVICES=0,1,2,4 python  main.py \
    --model mistralai/Mistral-7B-v0.1 \
    --cache_dir llm_weights/ \
    --prune_method sparsegpt_OWL \
    --sparsity_ratio 0.5 \
    --save results/mistral-OWL-0.5 \
    --ww_metric OWL_prune_ratio_mistral_0.5 \
    --ww_metric_cache ./data/mistral-7b/ \
    --epsilon 0.1 \
    --eval_wikitext False \
    --eval_zero_shot