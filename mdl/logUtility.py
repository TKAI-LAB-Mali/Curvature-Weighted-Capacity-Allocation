import argparse
import json
import os
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from kronfluence.task import CausalLanguageModelingTask
from kronfluence.analyzer import Analyzer
from peft import get_peft_model, LoraConfig 

# from lib.eval import eval_ppl, eval_zero_shot
# from lib.esd_utils import get_esd_metrics

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_llm(model, cache_dir='llm_weights'):
    model = AutoModelForCausalLM.from_pretrained(
        model,
        torch_dtype=torch.float16,
        cache_dir=cache_dir,
        low_cpu_mem_usage=True,
        device_map='auto'
    )

    model.seqlen = 2048
    return model


def get_flops(path):
    per_layer_base_flops = []
    per_layer_lora_flops_scaled = []
    with open(os.path.join(path, 'mola_lora_summary.json'), 'r') as f:
        summary = json.load(f)
    
    per_layer = summary.get('per_layer', {})
    
    for key, val in per_layer.items():
        base_flops = 0
        lora_flops = 0
        for attn_head in val:
            base_flops += attn_head['base_flops_per_token']
            lora_flops += attn_head['lora_flops_per_token_scaled']
        per_layer_base_flops.append(base_flops)
        per_layer_lora_flops_scaled.append(lora_flops)
    
    return per_layer_base_flops, per_layer_lora_flops_scaled

def tokenize_function(examples, tokenizer, max_seq_length):
    # This tokenizes the 'sentence' text from the dataset
    # We truncate long sequences and pad short ones
    return tokenizer(
        examples['sentence'],
        truncation=True,
        padding='max_length',
        max_length=max_seq_length
    )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, help='model type')
    parser.add_argument('--seed', type=int, default=0, help='Seed for sampling the calibration data')
    parser.add_argument('--nsamples', type=int, default=128, help='Number of calibration samples')
    parser.add_argument('--cache_dir', default='llm_weights', type=str)
    parser.add_argument('--save', type=str, default=None, help='Path to save results')
    parser.add_argument('--save_model', type=str, default=None, help='Path to save the model')
    
    experts_path = '../Expert_Allocation/layerIF_Computation/outputs/layerIF_values/mistral-7B'
    flops_paths = "../Expert_Allocation/layerIF_outputs/mistral_mola_46810_224_glue_cola_all"

    per_layer_base_flops, per_layer_lora_flops_scaled = get_flops(flops_paths)


    # Define the model name we want to load
    model_name = 'mistralai/Mistral-7B-v0.1'

    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Load the model
    model = get_llm(model_name)

    # Define the Task
    # This tells kronfluence how to compute the loss for our model.
    task = CausalLanguageModelingTask()

    # Create the Analyzer
    analyzer = Analyzer(
        analysis_name='glue-cola', # A name for this analysis run
        model=model, # our PEFT model
        task=task   # The causal LM task we defined
    )

    # load dataset
    dataset = load_dataset('glue', 'cola')

    batch_size = 8
    max_seq_length = 256

    kwargs = {
        'tokenizer': tokenizer,
        'max_seq_length': max_seq_length
    }
    # Apply tokenize functoin to all splits of the dataset
    tokenized_datasets = dataset.map(
                            tokenize_function, 
                            batched=True,
                            fn_kwargs=kwargs # Pass the extra arguments here
                        )

    # set the format to PyTorch tensors
    tokenized_datasets.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

    # Create training dataloader
    train_dataloader = DataLoader(
        tokenized_datasets['train'],
        batch_size=batch_size,
        collate_fn=task.collate_fn, # Use the task's special collate function
        shuffle=True    # shuffle the training data
    )

    valid_dataloader = DataLoader(
        tokenized_datasets['valid'],
        batch_size=batch_size,
        collate_fn=task.collate_fn,
        shuffle=False
    )

    # Compute and cache factors for influence function
    # This is the most computationally heavy stop
    # It goes through all training data and pre-computes the building blocks.
    analyzer.fit_all_factors(
        factors_name='IF_factors',
        dataset=train_dataloader
    )

    # Compute all pairwise influence scores with the computed factors.
    analyzer.compute_pairwise_scores(
        scores_name="IF_scores",
        factors_name="IF_factors",
        query_dataset=valid_dataloader,
        train_dataset=train_dataloader,
        per_device_query_batch_size=1024,
    )
    
    # Load the scores with dimension `len(eval_dataset) x len(train_dataset)`.
    scores = analyzer.load_pairwise_scores(scores_name="my_scores")
    
    # IF value for individual validation sample can be obtained as,
    # sample_score = scores[index]

    lora_r = 8
    lora_alpha = 16
    lora_dropout = 0.05
    # lora_target_modules
    lora_target_modules = [
        'q_proj', 'v_proj', 'k_proj', 'o_proj',
        'gate_proj', 'down_proj', 'up_proj'
    ]

    # create the config object
    peft_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=lora_target_modules,
        task_type='CAUSAL_LM' # This is import for a model like Mistral
    )

    # create the PEFT model
    model = get_peft_model(model, peft_config)

    # Print the number of trainable parameters
    model.print_trainable_parameters()

    

    

if __name__ == '__main__':
    main()

    