import os
import numpy as np
import torch
import random
import argparse
import json
import math
import torch
import util





def get_rho_l(b, lambda_value, n,  eta, layer_qualities, k, minmax=0.55):
    denom = 2 * eta * (layer_qualities)**k
    values = (b - lambda_value) * n / denom
    return np.minimum(minmax, np.maximum(0.0, values))
    # return np.maximum(0.0, values)


def prune_old(n_l, b, eta, layer_qualities, k, sparsity, minmax, epsilon=0.1):
    checksum = 0.0
    lambda_value = 0.0

    rho_l = get_rho_l(b, lambda_value, n_l, eta, layer_qualities, k, minmax)
    checksum = np.sum(n_l * rho_l)

    print(f'initial checksum: {checksum}')

    if checksum >= sparsity:
        return lambda_value, rho_l
    else:

        def get_G(lambda_value):
            rho_l = get_rho_l(b, lambda_value, n_l, eta, layer_qualities, k, minmax)
            value = np.sum(n_l * rho_l) - sparsity
            return value
        
        lambda_min_init = -100.0
        lambda_max_init = 1.0
        lambda_min = 0
        lambda_max = 1.0
        t = 1
        while get_G(lambda_min) < 0:
            print(f"G({lambda_min}): {get_G(lambda_min)}")
            lambda_min = lambda_min_init * (1 + math.exp(1))**t
            t += 1
        print(f"lambda_min: {lambda_min}, t: {t}")


        t = 1
        while get_G(lambda_max) > 0:
            print(f"G({lambda_max}): {get_G(lambda_max)}")
            lambda_max = lambda_max_init * (1 + math.exp(1))**t

            if t > 100: 
                print(f"Warning: Unable to find upper lambda bound; budget may be too small.upper_lam={lambda_max}\n")
                triggered = True 
                break
            t += 1
            


        print(f"lambda_max: {lambda_max}, t: {t}")
        print("\n====Starting bisection search...====\n")
        lambda_value = (lambda_min + lambda_max) * 0.5

        gap = get_G(lambda_value)
        print(f"G({lambda_value}): {gap}, epsilon: {epsilon}")
        
        iteration = 0
        max_iter = 1000
        while abs(gap) > epsilon and iteration < max_iter:
            iteration += 1
            

            # gap = get_G(lambda_value)


            if gap > 0:
                lambda_min = lambda_value
            else:
                lambda_max = lambda_value
            
            lambda_value = 0.5 * (lambda_min + lambda_max)

            gap = get_G(lambda_value)
        print(f"lambda_value: {lambda_value}, g(lambda): {abs(gap)}")

        print(f"Converged at iteration {iteration} with lambda={lambda_value} and gap={abs(gap)}\n")

        if iteration >= max_iter:
            print(f"Warning: Convergence not reached within {max_iter} iterations. Final gap={abs(gap):.4f}")

        
        
        rho_l = get_rho_l(b, lambda_value, n_l, eta, layer_qualities, k, minmax)
        return lambda_value, rho_l


def prune(n_l, b, eta, layer_qualities, k, sparsity, minmax, epsilon=0.1):
    def get_G(lambda_val):
        # We define G(lambda) internally so it always uses the correct local variables
        denom = 2 * eta * (layer_qualities ** k)
        values = (b - lambda_val) * n_l / denom
        rho = np.minimum(minmax, np.maximum(0.0, values))
        return np.sum(n_l * rho) - sparsity

    # 1. Evaluate unconstrained state
    G_0 = get_G(0.0)
    print(f"Initial Unconstrained Gap G(0): {G_0:.2f}")

    if abs(G_0) <= epsilon:
        # Extremely rare: lambda=0 exactly hits the target
        return 0.0, get_rho_l(b, 0.0, n_l, eta, layer_qualities, k, minmax)

    # 2. Establish strict mathematical bounds
    if G_0 > 0:
        # We are pruning TOO MUCH (e.g., 6.9B > 3.8B). 
        # We must INCREASE lambda to reduce rho.
        # Absolute mathematical ceiling: at lambda = b, rho = 0.
        lambda_min = 0.0
        lambda_max = float(b)  
        print(f"Target requires less pruning. Bounding lambda between [{lambda_min}, {lambda_max}]")
    else:
        # We are pruning TOO LITTLE (G_0 < 0).
        # We must DECREASE lambda (make it negative) to increase rho.
        lambda_max = 0.0
        lambda_min = -1.0
        while get_G(lambda_min) < 0:
            lambda_min *= 2.0  # Exponential drop
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
            # Still pruning too much, need to increase lambda
            lambda_min = lambda_value
        else:
            # Still pruning too little, need to decrease lambda
            lambda_max = lambda_value
            
        lambda_value = (lambda_min + lambda_max) / 2.0
        iteration += 1

    print(f"Converged at iteration {iteration} with lambda={lambda_value} and gap={abs(gap):.4f}\n")

    # 4. Return final computed values
    denom = 2 * eta * (layer_qualities ** k)
    values = (b - lambda_value) * n_l / denom
    rho_l = np.minimum(minmax, np.maximum(0.0, values))
    
    return lambda_value, rho_l


def normalize_layer_qualities(q, method='mean', k=1):
    """
    Stabilizes Information Flow (q_l) scores to prevent numerical explosion/underflow 
    when raised to the power of k.
    """
    # 1. Safety net: IF scores must be strictly positive. 
    # Replace any zeros or negative anomalies with a tiny positive float.
    q = np.maximum(q, 1e-7)
    
    if method == 'mean':
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
    # print(f"Projected q^k spread -> Min: {min_denom:.2e}, Max: {max_denom:.2e}")
    
    return q_norm

def compute_depth_scale(q, method='linear', strength=0.5, p=1):
    """
    Artificially inflates the IF scores based on structural importance.
    """
    L = len(q)
    indices = np.arange(L)
    
    # Calculate how far through the network we are (0.0 to 1.0)
    depth_ratio = indices / (L - 1) 
    
    if method == 'linear':
        weights = 1.0 + strength * (1.0 - depth_ratio)**p
    elif method == 'exponential':
        weights = np.exp(strength * (1.0 - depth_ratio))
    elif method == 'u_curve':
        # Parabolic function: High at edges (0.0 and 1.0), low in the center (0.5)
        # The '4' ensures the maximum added penalty exactly equals 'strength'
        weights = 1.0 + strength * 4.0 * (depth_ratio - 0.5)**2
    else:
        weights = np.ones(L)
        
    print(f"Depth Penalty ({method}) applied.")
    print(f"Layer 0 weight: {weights[0]:.2f}")
    print(f"Layer {L//2} (Middle) weight: {weights[L//2]:.2f}")
    print(f"Layer {L-1} weight: {weights[-1]:.2f}\n")
    
    return q * weights





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
        # q = normalize_layer_qualities(q, method='mean', k=k_value)
        penalty_method = 'linear'
        strength = 10.0
        p = 10.0
        # q = compute_depth_scale(q, method=penalty_method, strength=strength, p=p)

        sparsity_target = 0.5 * total_params
        minmax = 0.55
        minmax_array = np.ones_like(q) * minmax
        minmax_array[0:1] = 0.3
        minmax_array[1:4] = 0.4
        bits = 16

        lambda_value, rho_l = prune(n_l, bits, eta=2, 
                                    layer_qualities=q, #q_normalized,
                                      k=k_value, sparsity=sparsity_target, 
                                      minmax=minmax_array, epsilon=0.1)
        
        # lambda_value, rho_l = prune_old(n_l, bits, eta=2, 
        #                             layer_qualities=q, #q_normalized,
        #                               k=k_value, sparsity=sparsity_target, 
        #                               minmax=minmax, epsilon=0.1)

        os.makedirs('./data/', exist_ok=True, mode=0o755)
        if all(minmax_array == minmax):
            file_name = f"rho_l_{model_name}_{dataset}_mdl_minmax_{minmax}.json"
        else:
            array_desc = describe_array(minmax_array, name="", include_last_segment=False)
            file_name = f"rho_l_{model_name}_{dataset}_mdl_minmax_{minmax}{array_desc}.json"

            
        file_path = os.path.join('./data/', file_name)
        
        # os.makedirs('./data/', exist_ok=True, mode=0o755)
        # file_path = os.path.join('./data/', f"rho_l_{model_name}_{dataset}_mdl_minmax_{minmax}[0:3]_0_3.json")

        # depth_penalty_subdir = './data/depth_penalty/'
        # os.makedirs(depth_penalty_subdir, exist_ok=True, mode=0o755)
        # file_path = os.path.join(depth_penalty_subdir, f"rho_l_{model_name}_{dataset}_mdl_{penalty_method}_strength_{strength}_p_{p}_minmax_{minmax}[0:3]_0_3.json")

        # with open(file_path, "w") as f:
        #     json.dump({
        #         "lambda": lambda_value,
        #         "rho_l": rho_l.tolist(), 
        #         "target_sparsity": sparsity_target,
        #         "computed_sparsity": np.sum(n_l * rho_l),
        #         "total_params": total_params,
        #         "minmax": minmax_array.tolist(),
        #         # "strength": strength,
        #         # "p": p,

        #     }, f, indent=4)



if __name__ == "__main__":
    main()
