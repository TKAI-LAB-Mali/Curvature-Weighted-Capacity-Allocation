"""
Implementation of Alg-3 from overleaf
"""


import os
import numpy as np
import torch
import random
import argparse
import json
import math
import torch
import util

def calculate_retained_metrics(json_file, sparsity_ratio=0.3, b=16,
                                bit_precision=16):
    """
    we get value of 'b' by reading the number of flops per token obtained from stored .json files in layerIF_outputs folder. It contains dataset specific forward pass information for a model.
    
    :param json_file: path to json file
    :param sparsity_ratio: ratio of weights to be freezed
    :param bit_precision: precision of variable storing weight values
    """
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    unique_base_modules = set()
    per_layer_params = {}
    total_base_params = 0
    total_base_flops = 0
    total_lora_params = 0
    total_lora_flops = 0

    # Iterate through all layers and adapters
    for layer_idx, items in data.get('per_layer', {}).items():
        for item in items:
            module_name = item['module'] 

            # 1. Base Model Stats (Count ONLY ONCE per module name)
            # The JSON lists the module multiple times (once per adapter),
            # so we use a set to avoid double-counting the base weights.
            if module_name not in unique_base_modules:
                unique_base_modules.add(module_name)
                # Params = rows * cols

                #TO-DO: @Theophilus please check following step for getting n_l in the objective function 
                base_params = item['in_features'] * item['out_features']
                per_layer_params[layer_idx] =base_params    # n_l from the algorithm number of parameters


                # total_base_params += base_params
                # total_base_flops += item['base_flops_per_token']

            # 2. LoRA Stats (Count ALL, as they are additive)
            # LoRA Params = (A_rows * A_cols) + (B_rows * B_cols)
            # lora_p = (item['A_shape'][0] * item['A_shape'][1]) + \
            #          (item['B_shape'][0] * item['B_shape'][1])
            # total_lora_params += lora_p
            # total_lora_flops += item['lora_flops_per_token']

    # 3. Apply Sparsity
    # We only prune the base model, not the LoRA adapters
    # retained_base_params = b * total_base_params * (1 - sparsity_ratio)
    # retained_base_flops = b * total_base_flops * (1 - sparsity_ratio)

    # 4. Final Totals
    # final_params = retained_base_params + total_lora_params
    # final_flops = retained_base_flops + total_lora_flops
    # final_bits = final_params * bit_precision

    # print(f"-- Results for Sparsity {sparsity_ratio} ---")
    # print(f"Original Base Params: {total_base_params:,}")
    # print(f"Retained Base Params: {int(retained_base_params):,}")
    # print(f"LoRA Params (Added): {total_lora_params:,}")
    # print(f"TOTAL Retained Params: {int(final_params):,}")
    # print(f"TOTAL Model Bits: {int(final_bits):,} bits")
    # print(f'TOTAL Retained FLOPs: {int(final_flops):,}')

    return retained_base_params


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_rho_l(b, lambda_value, layer_params, eta, layer_quality, k):
    value = (b - lambda_value) * layer_params / (2 * eta * layer_quality**k)
    return min(1.0, max(0.0, value))

def prune(n_l, b, eta, layer_qualities, k, sparsity, epsilon=0.1):
    checksum = 0.0
    lambda_value = 0.0
    rho_l = [0.0] * len(n_l)
    for layer in range(len(n_l)):
        rho_l[layer] = get_rho_l(b, lambda_value, 
                                n_l[layer],
                                eta, layer_qualities[layer],
                                k)
        checksum += n_l[layer] * rho_l[layer]
    
    print(f'initial checksum: {checksum}')
    
    if checksum >= sparsity:
        return lambda_value, rho_l
    else:
        
        def get_G(temp_lambda):
            value = 0.0
            for layer in range(len(n_l)):
                value += n_l[layer] * get_rho_l(b, temp_lambda, 
                                                n_l[layer],
                                                eta, layer_qualities[layer],
                                                k)
            
            value -= sparsity
            # print(f"sum n_l*rho_l: {value}, sparsity: {sparsity}")
            return value

        lambda_min = -1.0
        lambda_max = 1.0
        t = 1
        while get_G(lambda_min) < 0:
            print(f"G({lambda_min}): {get_G(lambda_min)}")
            lambda_min = -1.0 * (1 + math.exp(1))**t
            t += 1
        print(f"lambda_min: {lambda_min}, t: {t}")
        t = 1
        while get_G(lambda_max) > 0:
            lambda_max = 1.0 * (1 + math.exp(1))**t
            t += 1
        print(f"lambda_max: {lambda_max}, t: {t}")
        lambda_value = (lambda_min + lambda_max) / 2.0
        g_lambda = get_G(lambda_value)
        print(f"g(lambda): {g_lambda}, epsilon: {epsilon}")
        while abs(g_lambda) > epsilon:
            # print(f"g(lambda): {g_lambda}, epsilon: {epsilon}")
            lambda_value = (lambda_min + lambda_max) / 2.0
            g_lambda = get_G(lambda_value)
            if g_lambda > 0:
                lambda_min = lambda_value
            else:
                lambda_max = lambda_value
            
            g_lambda = abs(get_G(lambda_value))
        print(f"lambda_value: {lambda_value}, g(lambda): {g_lambda}")
        for layer in range(len(n_l)):
            rho_l[layer] = get_rho_l(b, lambda_value, 
                                n_l[layer],
                                eta, layer_qualities[layer],
                                k)
        print(f"rho_l:")
        print(rho_l)
        return lambda_value, rho_l

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default="mistralai/Mistral-7B-v0.1", type=str)
    parser.add_argument('--bits', '-b', default=16, type=int, help='number of bits per weight value')
    parser.add_argument('--eta', type=float, default=0.1, help='eta in objective function')
    parser.add_argument('--rho', type=float, default=0.3, help='rho is sparsity level in a layer')
    parser.add_argument('--seed', type=int, default=0, help='Seed for sampling the calibration data')
    parser.add_argument('--save', type=str, default=None, help='Path to save results')
    parser.add_argument('--save_model', type=str, default=None, help='Path to save the model')
    args = parser.parse_args()

    # b -> bits/weight value. Currently we assume this to be 16 given weights are of type torch.bfloat16.

    experts_per_layer = [2,9,7,7,6,7,6,5,7,4,4,3,8,4,5,4,2,5,2,2,3,1,3,7,2,1,7,8,8,7,7,7] # [1,2,3]

    model_metadata = '/data/mdl-layerIF/Expert_Allocation/layerIF_outputs/mistral_mola_46810_224_glue_cola_all/mola_lora_summary.json'
    # retained_base_params = calculate_retained_metrics(model_metadata, sparsity_ratio=0.3)

    connections_per_layer = util.get_all_layer_connections(model_metadata)
    print(f"connection_per_layer: {connections_per_layer}")
    sparsity_target = sum(connections_per_layer) * 0.4
    lambda_value, rho_l = prune(n_l=connections_per_layer,
                                b=args.bits,
                                eta=args.eta, 
                                layer_qualities=util.get_IF(), 
                                k=1, 
                                sparsity=sparsity_target, 
                                epsilon=0.2)

