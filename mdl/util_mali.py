import os
import pickle as pkl
import numpy as np

def get_IF(
    experts_path: str,
    dataset: str,
    positive: bool = False,
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
                layer_score = float(np.mean(x))
            else:
                layer_score = float(np.sum(x))

            layer_IFs.append(layer_score)

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
