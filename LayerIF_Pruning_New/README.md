

## Installation
Create the conda environment from the file full-environment.yml

Then for mistral go to the folder lm-evaluation-harness and run:

```pip install -e .
   pip install torch==1.10.1+cu113
```

For gemma rename the lm-evaluation-harness-gemma folder to lm-evaluation-harness, then go to the folder lm-evaluation-harness and run:

```pip install -e .
   pip install torch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
```

## Usage
the run_xxx.sh files contain code to evaluate the models

Below is an example command for pruning Mistral-7B-v0.1 using Magnitude with our layerwise pruning ratios, to achieve unstructured 50% sparsity.
```
CUDA_VISIBLE_DEVICES=0,1,2,3 python  main.py \
    --model mistralai/Mistral-7B-v0.1 \
    --cache_dir llm_weights/ \
    --prune_method magnitude_ww \
    --sparsity_ratio 0.5 \
    --save results/mistral-IF-smoothed-0.5 \
    --ww_metric IF-300-96-smoothed \
    --ww_metric_cache ./data/mistral-7b/ \
    --epsilon 0.3 \
    --eval_wikitext False \
    --eval_zero_shot
```
We provide a quick overview of the arguments:  
- `--model`: The identifier for the LLaMA model on the Hugging Face model hub.
- `--cache_dir`: Directory for loading or storing LLM weights. The default is `llm_weights`.
- `--prune_method`: We have implemented these pruning methods, namely [`magnitude`, `wanda`, `sparsegpt`, `magnitude_ww`, `wanda_ww`, `sparsegpt_ww`].
- `--sparsity_ratio`: Denotes the percentage of weights to be pruned.
- `--save`: Specifies the directory where the result will be stored.
- `--ww_metric`: The ESDs metric used to diagnose layer quality and assign layer-wise pruning ratios. The default is `alpha_peak`.
- `--ww_metric_cache`: Directory for loading the layer-wise metric value of the model before pruning.
- `--epsilon`: Denotes the hyperparameter to control the range of pruning ratios. 

**To prune other LLMs, you can change the "model" argument in the script.**

**You can add data for different methods on the data folder. For OWL we computed the pruning ratio from their repository and added the ratios here manually**

