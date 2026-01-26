import json
import numpy as np
import os

def load_lora_costs(mola_path):
    """
    Returns per-layer LoRA FLOPs cost c_l.
    """
    with open(os.path.join(mola_path, "mola_lora_summary.json"), "r") as f:
        summary = json.load(f)

    per_layer = summary["per_layer"]

    c_l = []
    for layer_idx in sorted(per_layer.keys(), key=int):
        layer_cost = 0.0
        for item in per_layer[layer_idx]:
            layer_cost += item["lora_flops_per_token_scaled"]
        c_l.append(layer_cost)

    return np.array(c_l)

# Now to use do this Theo and/or Hitesh, in your logutility main function

# Load per-layer LoRA costs --> You get this from that json file, I assume it gives values 
c = load_lora_costs(flops_path)

# Capacity ratio (MDL knob) -- Just a hyperparameter
rho = 0.25        # try 0.1, 0.25, 0.5

# Budget --> This will give us budget
B = rho * np.sum(c)
