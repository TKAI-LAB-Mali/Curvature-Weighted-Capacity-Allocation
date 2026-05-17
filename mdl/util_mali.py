import os
import math
import pickle as pkl
import numpy as np
import json

def get_IF(
    experts_path: str,
    dataset: str,
    choice: str = "all",       # "all", "pos", or "neg"
    agg: str = "sum",          # "sum" or "mean"
    tau_frac: float = 0.05,    # prior strength (0.0 disables prior)
    eps: float = 1e-12
):
    """
    MDL-aligned LayerIF -> q_l conversion.

    Assumption (standard for influence): more NEGATIVE influence => more IMPORTANT.
    So we map: q_l = max(-IF_l, 0).

    We optionally add a small prior tau to avoid degeneracy when IFs are tiny/noisy:
        q_l <- q_l + tau
    where tau = tau_frac * median(q_l | q_l>0) (fallback to median(|IF|) if all zero).
    """

    layer_IFs = []

    lengths = {}
    layer = 0

    for file in sorted(os.listdir(experts_path)):
        if file.endswith(dataset + ".pkl"):
            with open(os.path.join(experts_path, file), "rb") as f:
                results = pkl.load(f)

            x = results["influence"]["proposed"].to_numpy()

            # Aggregate within this layer file
            # NOTE: do NOT keep only positives; do NOT invert.
            if agg == "mean":
                aggregated_score = np.mean(x, axis=0, 
                                           keepdims=True)
            elif agg == 'sum':
                aggregated_score = np.sum(x, axis=0, 
                                          keepdims=True)
                
            length_of_positive = np.sum(aggregated_score >= 0)
            length_of_negative = np.sum(aggregated_score < 0)

            lengths[layer] = (length_of_positive.item(), length_of_negative.item())
            layer += 1
                
            if choice == 'pos':
                aggregated_score = np.sum(aggregated_score[
                                                np.where(aggregated_score>=0)
                                                ])
                # aggregated_score = np.sum(aggregated_score[aggregated_score>=0])
            elif choice == 'neg':
                aggregated_score = np.sum(aggregated_score[
                                                np.where(aggregated_score<0)
                                                ])
                # aggregated_score = np.sum(aggregated_score[aggregated_score<0])
            else:  # all
                # print("==============================================================================")
                # print(f"Using choice {choice} for dataset {dataset}, layer {layer-1}\n")
                # print("==============================================================================")
                aggregated_score = np.sum(aggregated_score)

            layer_IFs.append(float(aggregated_score))

    layer_IFs = np.array(layer_IFs, dtype=float)
    # print(f"Raw Layer IFs for {dataset}: {layer_IFs}")

    # --- MDL-aligned mapping: negative => important ---
    q = np.maximum(-layer_IFs, 0.0)
    # q = np.max(layer_IFs) - layer_IFs

    # print(f"q after negation for {dataset}: {q}")

    # --- Optional prior to avoid collapse (MDL + capacity prior) ---
    tau_frac = 0.0
    if tau_frac > 0:
        nz = q[q > 0]
        if nz.size > 0:
            scale = np.median(nz)
        else:
            # fallback: use magnitude scale of raw IFs so tau isn't ~0
            scale = np.median(np.abs(layer_IFs)) + eps
        tau = tau_frac * (scale + eps)
        q = q + tau

    # keep strictly nonnegative
    q = np.maximum(q, 0.0)
    # print(f"q after scaling for {dataset}: {q}")

    # Helpful debug prints (remove later)
    # print(dataset, "IF min/max", layer_IFs.min(), layer_IFs.max(),
    #       "q min/max", q.min(), q.max(), "q>0", (q>0).sum(), '\n')

    return q, lengths
    


def solve_expert_allocation(
    q: np.ndarray,      # Layer qualities (q_l)  (assumed >= 0)
    c: np.ndarray,      # Costs per expert (c_l) (assumed > 0)
    alpha: np.ndarray,  # Coding slopes (alpha_l) (assumed > 0)
    B: float = 160,     # Budget constraint
    gamma: float = 0.9,
    beta: float = 2,
    epsilon: float = 0.1,
    max_iter: int = 1000
):
    """
    MDL-Optimal Expert Allocation via 1-D bisection on lambda.

    Implements the paper form:
        extra_e_l(lambda) = max( gamma*q_l^beta/(alpha_l + lambda*c_l) - 1, 0 )
        total_e_l(lambda) = 1 + extra_e_l(lambda)
    which is equivalent to total_e_l(lambda) = max(raw, 1) but clearer.

    Returns:
        (e_star_total_per_layer, lambda_star)
    """

   
    q = np.maximum(q, 0.0)
    c = np.maximum(c, 1e-12)
    alpha = np.maximum(alpha, 1e-12)

   
    def get_experts(lam: float) -> np.ndarray:
        # denom = alpha + (lam * c)
        denom = (alpha + lam) * c
        denom = np.maximum(denom, 1e-12)  # safety

        raw = gamma * (q ** beta) / denom

        # Paper form: extra then baseline
        extra = np.maximum(raw - 1.0, 0.0)
        total = 1.0 + extra
        return total

    
    def get_gap(lam: float) -> float:
        e_vals = get_experts(lam)
        if lam == 0.0:
            print(f"At lambda=0, raw experts = {e_vals}\n")
        return float(np.sum(c * e_vals) - B)

    if get_gap(0.0) <= epsilon:
        return get_experts(0.0), 0.0
    
    # print(f"Initial Experts at lambda=0: {get_experts(0.0)}\n")
    
    print(f"Gap at lambda=0 {get_gap(0.0)} <= {epsilon};\n")

    
    lower_lam = -1.0
    upper_lam = 1.0

    t=0
    while get_gap(lower_lam) < 0:
        lower_lam *= 2.0
        t += 1
        if t > 100: 
            print(f"Warning: Unable to find lower lambda bound; budget may be too large.lower_lam={lower_lam}\n")
            break

   
    t = 0
    triggered = False
    while get_gap(upper_lam) > 0:
        upper_lam *= 2.0
        t += 1
        if t > 100: 
            print(f"Warning: Unable to find upper lambda bound; budget may be too small.upper_lam={upper_lam}\n")
            triggered = True 
            break

    if not triggered:
        print(f"Found upper lambda bound: {upper_lam} with gap {get_gap(upper_lam)} and t = {t}\n")
    

    print(f"Initial lambda bounds: lower={lower_lam}, upper={upper_lam}\n")
    # --- 5) Bisection ---
    current_lam = 0.5 * (lower_lam + upper_lam)
    gap = get_gap(current_lam)
    print(f"Initial gap at lambda={current_lam}: {gap}\n")

    # print(f"Experts at lambda={current_lam}: {get_experts(current_lam)}\n")

    iteration = 0
    while abs(gap) > epsilon and iteration < max_iter:
        iteration += 1

        if gap > 0:
            # Cost > Budget: increase lambda to reduce experts
            lower_lam = current_lam
        else:
            # Cost < Budget: decrease lambda
            upper_lam = current_lam

        current_lam = 0.5 * (lower_lam + upper_lam)
        gap = get_gap(current_lam)

    print(f"Converged at iteration {iteration} with lambda={current_lam} and gap={gap}\n")

    if iteration >= max_iter:
        print(f"Warning: Convergence not reached within {max_iter} iterations. Final gap={gap:.4f}")

    
    return get_experts(current_lam), current_lam



def get_layer_connections(mola_path):
    """
    Returns per-layer LoRA FLOPs cost c_l.
    """
    with open(os.path.join(mola_path, "model_summary.json"), "r") as f:
        summary = json.load(f)

    per_layer = summary["per_layer"]

    n_l = []
    
    for layer_idx in sorted(per_layer.keys(), key=int):
        layer_param = 0.0
        for item in per_layer[layer_idx]:
            
            params = item['in_features'] * item['out_features']
            layer_param += params

        n_l.append(layer_param)



    return np.array(n_l),  np.sum(n_l)