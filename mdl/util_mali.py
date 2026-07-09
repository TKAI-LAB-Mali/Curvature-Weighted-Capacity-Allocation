import os
import math
import pickle as pkl
import numpy as np

def get_IF(
    experts_path: str,
    dataset: str,
    choice: str,               # 'pos', 'neg', 'all'
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
                
            if choice == 'pos':
                aggregated_score = np.sum(aggregated_score[
                                                np.where(aggregated_score>=0)
                                                ])
            elif choice == 'neg':
                aggregated_score = np.sum(aggregated_score[
                                                np.where(aggregated_score<0)
                                                ])
            elif choice == 'all':
                aggregated_score = np.sum(aggregated_score)
                
            layer_IFs.append(aggregated_score)

    layer_IFs = np.array(layer_IFs, dtype=float)

    # --- MDL-aligned mapping: negative => important ---
    q = np.maximum(-layer_IFs, 0.0)

    # --- Optional prior to avoid collapse (MDL + capacity prior) ---
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

    # Helpful debug prints (remove later)
    # print(dataset, "IF min/max", layer_IFs.min(), layer_IFs.max(),
    #       "q min/max", q.min(), q.max(), "q>0", (q>0).sum())

    return q


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
        denom = alpha + (lam * c)
        denom = np.maximum(denom, 1e-12)  # safety

        raw = gamma * (q ** beta) / denom

        # Paper form: extra then baseline
        extra = np.maximum(raw - 1.0, 0.0)
        total = 1.0 + extra
        return total

    
    def get_gap(lam: float) -> float:
        e_vals = get_experts(lam)
        return float(np.sum(c * e_vals) - B)

    if get_gap(0.0) <= epsilon:
        return get_experts(0.0), 0.0

    
    lower_lam = 0.0
    upper_lam = 1.0

   
    t = 0
    while get_gap(upper_lam) > 0:
        upper_lam *= 2.0
        t += 1
        if t > 80:  
            break

    # --- 5) Bisection ---
    current_lam = 0.5 * (lower_lam + upper_lam)
    gap = get_gap(current_lam)

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

    if iteration >= max_iter:
        print(f"Warning: Convergence not reached within {max_iter} iterations. Final gap={gap:.4f}")

    
    return get_experts(current_lam), current_lam
