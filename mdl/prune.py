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
    eta = 2
    bits = 16
    kappa = 2
    
    for dataset in datasets:


        q, lengths = mutil.get_IF(influence_path, dataset, choice)
        print(f"Dataset: {dataset}, IF min/max: {q.min()}/{q.max()}, q>0: {(q>0).sum()}\n") 
        n_l, total_params = mutil.get_layer_connections("/data/mdl-layerIF/mdl")
     

        sparsity_target = 0.5 * total_params
        minmax = 0.55
        minmax_array = np.ones_like(q) * minmax
        

        lambda_value, rho_l = prune(n_l, bits, eta=eta, 
                                    layer_qualities=q, #q_normalized,
                                      k=kappa, sparsity=sparsity_target, 
                                      minmax=minmax_array, epsilon=0.1)
        

        os.makedirs('./data/', exist_ok=True, mode=0o755)
        if all(minmax_array == minmax):
            file_name = f"rho_l_{model_name}_{dataset}_mdl_minmax_{minmax}.json"
        else:
            array_desc = describe_array(minmax_array, name="", include_last_segment=False)
            file_name = f"rho_l_{model_name}_{dataset}_mdl_minmax_{minmax}{array_desc}.json"

        directory = f'./data/kappa_{int(kappa)}/'  
        os.makedirs(directory, exist_ok=True, mode=0o755)
        file_path = os.path.join(directory, file_name)
        
        # os.makedirs('./data/', exist_ok=True, mode=0o755)
        # file_path = os.path.join('./data/', f"rho_l_{model_name}_{dataset}_mdl_minmax_{minmax}[0:3]_0_3.json")

        

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
                "minmax": minmax_array.tolist(),
                # "strength": strength,
                # "p": p,

            }, f, indent=4)
        print(f"rho_l for {dataset} is {rho_l}\n")



if __name__ == "__main__":
    main()
