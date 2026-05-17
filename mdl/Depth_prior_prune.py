import os
import numpy as np
import torch
import random
import argparse
import json
import math
import torch
import util



def cal_bit_cost(tau_sign, b, tau, d, lambda_value=0.0):
    if tau_sign == 'positive':
        if lambda_value == 0.0:
            return b + tau * d
        return b + lambda_value + tau * d
    elif tau_sign == 'negative':
        if lambda_value == 0.0:
            return b - tau * d
        return b + lambda_value - tau * d
    else:
        raise ValueError("tau_sign must be 'positive' or 'negative'")
    
def get_rho_l(b, d, tau, lambda_value, n,  eta, layer_qualities, k, minmax=0.55):
    denom = 2 * eta * (layer_qualities)**k
    numerator = b + lambda_value + tau * d
    values = numerator / denom
    return np.minimum(minmax, np.maximum(0.0, values))
    # return np.maximum(0.0, values)


def dfunc(Length):
    d = (1 - np.arange(1, Length+1)/(Length-1)) - 1/Length * np.sum(1 - np.arange(1, Length+1)/(Length-1))
    return d

def prune(n_l, b, eta, layer_qualities, k, tau, sparsity, minmax, epsilon=0.1):
    # layer_qualities = layer_qualities/n_l
    d = dfunc(len(n_l))
    def get_G(lambda_val):
        denom = 2 * eta * (layer_qualities ** k)
        numerator = b + lambda_val + tau * d
        values = numerator / denom
        rho = np.minimum(minmax, np.maximum(0.0, values))
        return np.sum(n_l * rho) - sparsity

    # 1. Evaluate unconstrained state
    print("Evaluating unconstrained pruning (lambda=0)...")
    G_0 = get_G(0.0)
    print(f"Initial Unconstrained Gap G(0): {G_0:.2f}")

    if abs(G_0) <= epsilon:
        lambda_value = 0.0
        return lambda_value, get_rho_l(b, d, tau, lambda_value, n_l, eta, layer_qualities, k, minmax)

    # 2. Establish strict mathematical bounds
    if G_0 > 0:
        # We PRUNED TOO MANY parameters (G > 0). 
        # We must DECREASE lambda to reduce rho.
        lambda_max = 0.0
        max_baseline = np.max(b + tau * d)
        lambda_min = -float(max_baseline)  # This ensures rho can be zero if needed.
        print(f"Target requires less pruning. Bounding lambda between [{lambda_min}, {lambda_max}]")
    else:
        # We PRUNED TOO FEW parameters (G < 0).
        # We must INCREASE lambda to increase rho.
        lambda_min = 0.0
        lambda_max = 1.0
        while get_G(lambda_max) < 0:
            lambda_max *= 2.0  
            if lambda_max > 1e30:
                raise OverflowError(f"Cannot reach target sparsity {sparsity}. Check if minmax limits restrict rho too heavily.")
        print(f"Target requires more pruning. Bounding lambda between [{lambda_min}, {lambda_max}]")

    
    print("\n==== Starting Bisection Search ====\n")
    iteration = 0
    max_iter = 1000
    lambda_value = (lambda_min + lambda_max) / 2.0

    # 3. Execute Bisection
    while iteration < max_iter:
        gap = get_G(lambda_value)
        
        if abs(gap) <= epsilon:
            break
            
        if gap > 0:
            # We are ABOVE the target gap (pruning too much).
            # We must DECREASE lambda.
            lambda_max = lambda_value
        else:
            # We are BELOW the target gap (pruning too little).
            # We must INCREASE lambda.
            lambda_min = lambda_value
            
        lambda_value = (lambda_min + lambda_max) / 2.0
        iteration += 1

    print(f"Converged at iteration {iteration} with lambda={lambda_value} and gap={abs(gap):.4f}\n")

    # 4. Return final computed values
    denom = 2 * eta * (layer_qualities ** k)
    numerator = b + lambda_value + tau * d
    values = numerator / denom
    rho_l = np.minimum(minmax, np.maximum(0.0, values))
    
    return lambda_value, rho_l

def normalize_layer_qualities(q, method='mean', k=1):
    """
    Stabilizes Information Flow (q_l) scores to prevent numerical explosion/underflow 
    when raised to the power of k.
    """
    # 1. Safety net: IF scores must be strictly positive. 
    # Replace any zeros or negative anomalies with a tiny positive float.
    # q = np.maximum(q, 1e-7)
    if method == '':
        # No normalization. Use raw q_l values.
        q_norm = q
    
    elif method == 'mean':
        # Centers the average quality exactly at 1.0.
        # Best for low-to-medium k values (k=1 to k=5).
        q_norm = q / np.mean(q)
        
    elif method == 'bounded':
        # Maps q strictly into the range [0.5, 1.5].
        # CRITICAL for extreme k values (like k=20) to prevent float overflow.
        q_min, q_max = np.min(q), np.max(q)
        q_scaled = (q - q_min) / (q_max - q_min + 1e-9) # Scale to [0, 1]
        q_norm = 0.5 + q_scaled                         # Shift to [0.5, 1.5]
        
    elif method == 'log_smooth':
        # If your IF scores have massive outlier layers (e.g., one layer is 1000x others),
        # this dampens the extreme outliers before taking the mean.
        q_log = np.log1p(q)
        q_norm = q_log / np.mean(q_log)
        
    print(f"Normalized q_l (Method: {method}) | Min: {q_norm.min():.4f}, Max: {q_norm.max():.4f}, Mean: {np.mean(q_norm):.4f}")
    
    # Optional: Print a warning if the resulting denominator will still explode
    max_denom = np.max(q_norm)**k
    min_denom = np.min(q_norm)**k
    print(f"Projected q^k spread -> Min: {min_denom:.2e}, Max: {max_denom:.2e}")
    
    return q_norm





def main():


    import util_mali as mutil
    import json
    from util import describe_array
    model_name = 'gemma-7b'
    influence_path = f"/data/mdl-layerIF/Expert_Allocation/LayerIF_Computation/outputs/layerIF_values/{model_name}"
    file_attachment = "negative_IF_score_beta_3"
    choice = 'all'  # 'pos', 'neg', or 'all'
    datasets = ['mrpc', 'cola', 'openbook', 'text_science_q_rebuttal','commonq']
    # datasets = ['text_science_q_rebuttal']

    print(f"Datasets to process: {datasets}")
    filed_Experts = ''
    printed_Experts = ''
    
    for dataset in datasets:


        q, lengths = mutil.get_IF(influence_path, dataset, choice)
        print(f"Dataset: {dataset}, IF min/max: {q.min()}/{q.max()}, q>0: {(q>0).sum()}\n") 
        n_l, total_params = mutil.get_layer_connections("/data/mdl-layerIF/mdl")
        print(f"per layer param : {n_l}\n total param: {total_params}\n")

        k_value = 2
        q = (q / n_l) + 1e-8  # Normalize IF by layer parameter count
        q_norm_method = '_mean'  #"", '_mean', '_bounded', or '_log_smooth'
        q = normalize_layer_qualities(q, method=q_norm_method[1:], k=k_value)
        

        sparsity_target = 0.5 * total_params
        minmax = 0.55
        minmax_array = np.ones_like(q) * minmax
        # minmax_array[0:4] = 0.3
        
        bits = 16
        tau = 4

        lambda_value, rho_l = prune(n_l, bits, eta=2, 
                                    layer_qualities=q, #q_normalized,
                                      k=k_value,tau=tau,  sparsity=sparsity_target, 
                                      minmax=minmax_array, epsilon=0.1)
        
        # lambda_value, rho_l = prune_old(n_l, bits, eta=2, 
        #                             layer_qualities=q, #q_normalized,
        #                               k=k_value, sparsity=sparsity_target, 
        #                               minmax=minmax, epsilon=0.1)
        
        os.makedirs('./data_dp/', exist_ok=True, mode=0o755)
        if all(minmax_array == minmax):
            file_name = f"rho_l_{model_name}_{dataset}_mdl_tau_{tau}{q_norm_method}_minmax_{minmax}.json"
        else:
            array_desc = describe_array(minmax_array, name="")
            file_name = f"rho_l_{model_name}_{dataset}_mdl_tau_{tau}{q_norm_method}_minmax_{minmax}{array_desc}.json"

            
        file_path = os.path.join('./data_dp/', file_name)

        # depth_penalty_subdir = './data/depth_penalty/'
        # os.makedirs(depth_penalty_subdir, exist_ok=True, mode=0o755)
        # file_path = os.path.join(depth_penalty_subdir, f"rho_l_{model_name}_{dataset}_mdl_{penalty_method}_strength_{strength}_p_{p}_minmax_{minmax}[0:3]_0_3.json")

        with open(file_path, "w") as f:
            json.dump({
                "lambda": lambda_value,
                "rho_l": rho_l.tolist(), 
                "target_sparsity": sparsity_target,
                "computed_sparsity": np.sum(n_l * rho_l),
                "total_params": total_params,
                "tau": tau,
                "quality_normalization": q_norm_method,
                "minmax": minmax_array.tolist(),

                # "strength": strength,
                # "p": p,

            }, f, indent=4)



if __name__ == "__main__":
    main()
