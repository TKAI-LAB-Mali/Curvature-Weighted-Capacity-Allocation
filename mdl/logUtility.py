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
    return max(value, 1)

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
            # print(f"abs(g_val): {abs(g_val)}")
            if g_val > 0:
                lambda_lb = lambda_avg
            else:
                lambda_ub = lambda_avg
        
        for layer in range(len(layer_quality)):
            e_l[layer] = float(get_el(layer_quality[layer], c_l[layer], lambda_avg))
        return lambda_avg, e_l
    


import numpy as np

def solve_expert_allocation(
    q: np.ndarray,      # Layer qualities (q_l)
    c: np.ndarray,      # Costs per expert (c_l)
    alpha: np.ndarray,  # Coding slopes (alpha_l)
    B: float = 160,           # Budget constraint
    gamma: float = 0.9,       # MDL hyperparameter
    beta: float = 2,        # MDL hyperparameter
    epsilon: float = 0.1, # Tolerance (epsilon)
    max_iter: int = 1000   # Safety break for infinite loops
):
    """
    Solves the MDL-Optimal Expert Allocation problem using bisection on lambda.
    
    Returns:
        tuple: (e_star, lambda_star)
    """
    
    # --- 1. Define closed form e_l(lambda) ---
    # Formula: max( (gamma * q^beta) / (alpha + lambda * c) - 1, 0 )
    def get_experts(lam):
        # We use np.maximum to enforce e >= 0
        # Adding a tiny epsilon to denominator avoids div-by-zero if alpha=0 & lam=0
        numerator = gamma * (q ** beta)
        denominator = alpha + (lam * c)
        
        # Calculate raw value, handling potential division issues gracefully
        with np.errstate(divide='ignore', invalid='ignore'):
            raw_e = (numerator / denominator) # - 1
            
        return np.maximum(raw_e, 1)

    # --- 2. Define constraint function g(lambda) ---
    # g(lambda) = Sum(c_l * e_l) - B
    def get_gap(lam):
        e_vals = get_experts(lam)
        total_cost = np.sum(c * e_vals)
        return total_cost - B

    # --- 3. Check Unconstrained Case (Step 3 in Algo) ---
    # If budget is satisfied with no penalty (lambda=0), we stop.
    if get_gap(0.0) <= 0:
        return get_experts(0.0), 0.0

    # --- 4. Initialize Bounds (Step 4.1 in Algo) ---
    # We need g(lower) >= 0 and g(upper) <= 0
    lower_lam = 0.0
    upper_lam = 1.0
    
    # Exponentially grow upper_lam until g(upper_lam) <= 0
    # This handles the case where a very high penalty is needed to meet budget
    t = 0
    while get_gap(upper_lam) > 0:
        # upper_lam *= 2.0
        # if upper_lam > 1e12: # Safety break for unreachable budgets
        #     raise ValueError("Budget is unattainable (upper_lam exploded).")
        upper_lam = (1.0 + math.exp(1))**t
        t += 1

    

    # --- 5. Bisection Search (Step 4.2 in Algo) ---
    # Initialize lambda
    current_lam = (lower_lam + upper_lam) / 2.0
    gap = get_gap(current_lam)
    
    iteration = 0
    while abs(gap) > epsilon:
        # Safety break
        iteration += 1
        if iteration > max_iter:
            print(f"Warning: Convergence not reached within {max_iter} iterations.")
            break
            
        # Update bounds
        if gap > 0:
            # Cost > Budget: Need higher penalty (lambda) to reduce cost
            lower_lam = current_lam
        else:
            # Cost < Budget: Can afford lower penalty
            upper_lam = current_lam
            
        # Update lambda and recalculate gap
        current_lam = (lower_lam + upper_lam) / 2.0
        gap = get_gap(current_lam)

    # --- 6. Return Results ---
    return get_experts(current_lam), current_lam




def solve_interval_allocation_robust(
    q: np.ndarray, c: np.ndarray, alpha: np.ndarray,
    B: float, k_fraction: float, gamma: float, beta: float,
    epsilon: float = 1e-4
):
    # Avoid zero division
    q = np.maximum(q, 1e-9)
    B_lower, B_upper = k_fraction * B, B
    
    # Singularity Limit: lambda cannot be lower than this (denominator becomes 0)
    # limit = max(-alpha / c). Since alpha, c > 0, this is a negative number.
    # We add a tiny buffer to avoid actual division by zero.
    limit = np.max(-alpha / c)
    
    def get_experts_and_cost(lam):
        # Denominator
        denom = alpha + (lam * c)
        
        # If we hit the singularity, cost is infinite
        if np.any(denom <= 1e-12):
            return np.inf

        # Raw Formula
        raw_e = (gamma * (q ** beta)) / denom
        
        # Apply strict floor of 1.0
        e_final = np.maximum(raw_e, 1.0)
        return np.sum(c * e_final)

    # --- 1. Check Baseline (Lambda = 0) ---
    cost_0 = get_experts_and_cost(0.0)
    
    if B_lower <= cost_0 <= B_upper:
        return get_experts_and_cost_vector(0.0, q, c, alpha, gamma, beta), 0.0

    # --- 2. Determine Search Range ---
    lam_low, lam_high = 0.0, 0.0
    
    if cost_0 > B_upper:
        # Too expensive -> Tax (Positive Lambda)
        lam_low = 0.0
        lam_high = 1.0
        t = 0
        # Exponential search upwards
        while get_experts_and_cost(lam_high) > B_upper:
            # lam_high *= 2.0
            # if lam_high > 1e12: break # Safety break
            lam_high = (1.0 + math.exp(1))**t
            t += 1
            
    else:
        # Too cheap -> Subsidy (Negative Lambda)
        # We need to find a lambda closer to 'limit' that triggers higher cost
        lam_high = 0.0
        
        # Start safely away from the singularity
        lam_low = limit * 0.5 
        
        # Aggressively push lam_low towards 'limit' until cost > B_lower
        # limit is negative, so "increasing factor" means moving toward 1.0
        factor = 0.9
        while get_experts_and_cost(lam_low) < B_lower:
            # Move closer to the cliff edge
            lam_low = limit * factor
            
            # Increase factor (0.9 -> 0.99 -> 0.999 ...)
            factor = 1.0 - (1.0 - factor) / 10.0
            
            # If we get too close to precision limits, stop
            if (1.0 - factor) < 1e-12:
                print("Warning: Max budget capacity reached (mathematical limit).")
                break

    # --- 3. Bisection ---
    current_lam = (lam_low + lam_high) / 2.0
    
    for _ in range(100):
        cost = get_experts_and_cost(current_lam)
        
        # Target depends on where we are
        if cost > B_upper:
            # Need to reduce cost -> Increase lambda (towards 0 or positive)
            lam_low = current_lam 
        elif cost < B_lower:
            # Need to increase cost -> Decrease lambda (towards singularity)
            lam_high = current_lam
        else:
            # Inside the interval! Success.
            break
            
        current_lam = (lam_low + lam_high) / 2.0

    # Return final vectors
    final_e = get_experts_and_cost_vector(current_lam, q, c, alpha, gamma, beta)
    return final_e, current_lam

def get_experts_and_cost_vector(lam, q, c, alpha, gamma, beta):
    denom = alpha + (lam * c)
    # Safety clamp for plotting/returning
    denom = np.maximum(denom, 1e-9) 
    raw_e = (gamma * (q ** beta)) / denom
    return np.maximum(raw_e, 1.0)


        

def main():
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--model', type=str, help='model type')
    # parser.add_argument('--seed', type=int, default=0, help='Seed for sampling the calibration data')
    # parser.add_argument('--nsamples', type=int, default=128, help='Number of calibration samples')
    # parser.add_argument('--cache_dir', default='llm_weights', type=str)
    # parser.add_argument('--save', type=str, default=None, help='Path to save results')
    # parser.add_argument('--save_model', type=str, default=None, help='Path to save the model')
    
    # experts_path = '/data/mdl-layerIF/Expert_Allocation/layerIF_Computation/outputs/layerIF_values/gemma-7b'
    influence_path = "/data/mdl-layerIF/Expert_Allocation/LayerIF_Computation/outputs/layerIF_values/gemma-7b"

    datasets = ['mrpc', 'cola', 'openbook', 'text_science_q_rebuttal','commonq']
    # datasets = ['commonq']

    print(f"Datasets to process: {datasets}")
    Experts = ''
    for dataset in datasets:
        print(f"Processing dataset: {dataset}")

    

        layerIFs = util.get_IF(influence_path, dataset, True)
        # print(f"Raw LayerIFs : {layerIFs}")
        budget = 160 #8 * len(layerIFs)
        # layer_quality = [math.sqrt(value) for value in layerIFs]
        # lambda_avg_old, e_l_old = logUtility(budget, layer_quality=layerIFs)
        e_l, lambda_avg = solve_expert_allocation(np.array(layerIFs),
                                                np.array([1.0]*len(layerIFs)),
                                                np.array([0.5]*len(layerIFs)),
                                                B=budget,
                                                gamma=0.9,
                                                beta=3,
                                                epsilon=0.1,
                                                max_iter=1000)
        e_l, _ = solve_interval_allocation_robust(np.array(layerIFs),
                                                np.array([1.0]*len(layerIFs)),
                                                np.array([0.5]*len(layerIFs)),
                                                B=budget,
                                                k_fraction=0.99,
                                                gamma=0.9,
                                                beta=2,
                                                epsilon=0.1)
        e_l = np.floor(np.array(e_l))
        e_l_sum = np.sum(e_l)
        e_l = e_l.astype(int).tolist()
        e_l = str(e_l).replace('[','').replace(']','').replace(' ', '')

        Experts += f"{dataset}: {e_l} := Sum : {e_l_sum}\n"

    

        # print(f"MDL Expert Allocation Results old: {np.array(lambda_avg_old)}, {np.floor(e_l_old)}")
        # print(f"MDL Expert Allocation Results new: {e_l}")
    print("Final Expert Allocations per dataset:")
    print(Experts)
    
    with open('MDL_allocated_experts_gemma_positive.txt', 'w') as f:
        f.write(Experts)

    # print(np.floor(np.array(e_l_old)) == np.floor(np.array(e_l)))
    # print(f"original e_l: {e_l}")
    # e_l = np.array([int(round(x)) for x in e_l])
    # print(f"rounded e_l: {e_l}")
    # print(f"sum(e_l): {sum(e_l)}, budget:{budget}")

    # number of experts per layer obtained from Hadi's code
    # layerIF_experts = np.array([1, 11, 9, 7, 8, 9, 9, 7, 8, 8, 8, 7, 6, 8, 6, 4, 7, 3, 3, 3, 4, 4, 3, 2, 5, 2, 1, 2, 1, 2, 1, 1])
    
    # check the difference between our number of experts and Hadi's values
    # assert len(e_l) == len(layerIF_experts)
    # error = np.sqrt(np.sum((layerIF_experts - e_l)**2) / len(e_l))
    # print(f"L2 error between MDL and layerIF prediction of experts: {error}")

if __name__ == '__main__':
    main()

    