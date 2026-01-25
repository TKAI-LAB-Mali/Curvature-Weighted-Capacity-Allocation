"""
Implementation of Alg-1 from overleaf
"""


import argparse
import json
import os
import random
import numpy as np
import torch
import math
import util
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
# from kronfluence.task import CausalLanguageModelingTask
# from kronfluence.analyzer import Analyzer
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

def get_el(layer_quality, c_l, lambda_value=1, beta=2, alpha=0.5, gamma=0.9):
    value = gamma * layer_quality**beta / (alpha + lambda_value * c_l)
    return max(value, 1.0)

def logUtility(budget: int, layer_quality: list):
    cost = 0
    lambda_value = 0.0
    c_l = [1.0] * len(layer_quality)
    e_l = [0.0] * len(layer_quality)
    for layer in range(len(layer_quality)):
        e_l[layer] = get_el(layer_quality[layer],
                    c_l[layer],
                    lambda_value)
        cost += c_l[layer] * e_l[layer]
    if cost <= budget:
        lambda_value = 0.0
        return lambda_value, e_l
    else:
        def get_g(lambda_val):
            value = 0
            for layer in range(len(layer_quality)):
                value += c_l[layer] * get_el(layer_quality[layer],
                                             c_l[layer],
                                             lambda_val)
            value -= budget
            return value
        
        lambda_lb = 0.0
        lambda_ub_init = 1.0
        lambda_ub = 1.0
        t = 0
        while get_g(lambda_ub) > 0:
            lambda_ub = lambda_ub_init * (1.0 + math.exp(1))**t
            t += 1
        # print(f"g(lambda_lb): {get_g(lambda_lb)}")
        # print(f"g(lambda_ub): {get_g(lambda_ub)}")

        epsilon = 0.1
        lambda_avg = (lambda_lb + lambda_ub) / 2.0
        g_val = get_g(lambda_avg)
        # print(f"lambda_avg: {lambda_avg}, g_val: {g_val}, epsilon: {epsilon}")
        # print(f"initial abs(g_val): {abs(g_val)}")
        while abs(g_val) > epsilon:
            lambda_avg = (lambda_lb + lambda_ub) / 2.0
            g_val = get_g(lambda_avg)
            if g_val > 0:
                lambda_lb = lambda_avg
            else:
                lambda_ub = lambda_avg
        
        for layer in range(len(layer_quality)):
            e_l[layer] = get_el(layer_quality[layer], c_l[layer], lambda_avg)
        return lambda_avg, e_l
        

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, help='model type')
    parser.add_argument('--seed', type=int, default=0, help='Seed for sampling the calibration data')
    parser.add_argument('--nsamples', type=int, default=128, help='Number of calibration samples')
    parser.add_argument('--cache_dir', default='llm_weights', type=str)
    parser.add_argument('--save', type=str, default=None, help='Path to save results')
    parser.add_argument('--save_model', type=str, default=None, help='Path to save the model')
    
    experts_path = '/data/mdl-layerIF/Expert_Allocation/layerIF_Computation/outputs/layerIF_values/Mistral-7B-v0.1'
    output_folder = '/data/mdl-layerIF/Expert_Allocation/layerIF_outputs/'
    data_paths = {
        '_cola': 'mistral_mola_46810_224_glue_cola_all',
        'mrpc': 'mistral_mola_46810_224_glue_mrpc_all',
        'commonq': 'mistral_mola_46810_224_qa_commonq_all',
        'openbook': 'mistral_mola_46810_224_qa_openbook_all',
        'text_science_q_rebuttal': "mistral_mola_46810_224_qa_text_scienceq_all"}

    # Program flow:
    # step 1 - implement Expert_Allocation/layerIF_Computation/compute_IF.py to obtain the IF scores for LLM (mention model name).
    # step 2 - implement expert_allocator.ipynb with the IF scores calculated in the previous step. This will give us the number of experts per layer.
    # step 3 - run this file to obtain MDL based function (Alg-1, log-utility in the draft) number of experts per layer

    for key, value in data_paths.items():
        path = os.path.join(output_folder, value)
        print(f"path: {path}")
        per_layer_base_flops, per_layer_lora_flops_scaled = get_flops(path)

        layerIFs = util.get_IF(experts_path, dataset=key)
        budget = 160 #8 * len(layerIFs)
        # layer_quality = [math.sqrt(value) for value in layerIFs]
        lambda_avg, e_l = logUtility(budget, layer_quality=layerIFs)
        # print(f"lambda_avg: {lambda_avg}, sum(e_l): {sum(e_l)}")
        # print(f"original e_l: {e_l}")
        e_l = [max(1, math.floor(x)) for x in e_l]
        print(f'exp_number_experts="{",".join(map(str, e_l))}"')
        top_k = [2 if e>1 else 1 for e in e_l]
        print(f'exp_top_k="{",".join(map(str, top_k))}"\n')
        

    # # number of experts per layer obtained from Hadi's code
    # layerIF_experts = np.array([1,1,1,1,2,2,4,6,4,6,6,6,6,6,7,7,7,7,6,3,7,6,6,5,5,6,7,6,5,6,7,5])
    
    # # check the difference between our number of experts and Hadi's values
    # assert len(e_l) == len(layerIF_experts)
    # error = np.sqrt(np.sum((layerIF_experts - e_l)**2) / len(e_l))
    # print(f"L2 error between MDL and layerIF prediction of experts: {error}")

if __name__ == '__main__':
    main()

    