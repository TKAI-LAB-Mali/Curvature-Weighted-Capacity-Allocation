import json
import numpy as np
import os

def load_lora_costs(mola_path):
    """
    Returns per-layer LoRA FLOPs cost c_l.
    """
    with open(os.path.join(mola_path, "model_summary.json"), "r") as f:
        summary = json.load(f)

    per_layer = summary["per_layer"]

    c_l = []
    b_l = []
    plm = []
    for layer_idx in sorted(per_layer.keys(), key=int):
        layer_cost = 0.0
        layer_budget = 0.0
        per_layer_modules = 0
        for item in per_layer[layer_idx]:
            layer_budget += item["base_flops_per_token"]
            layer_cost += item["lora_flops_per_token_scaled"]
            per_layer_modules += 1
        c_l.append(layer_cost)
        b_l.append(layer_budget)
        plm.append(per_layer_modules)
    
    # print("Per-layer modules:", plm)

    return np.array(c_l),  np.sum(b_l)

# Now to use do this Theo and/or Hitesh, in your logutility main function




# Below is proxy if using standard lora, which doesn't contain flops etc information

def compute_c_l_from_lora(
    model_config,
    lora_rank: int,
    lora_modules=("q_proj", "k_proj", "v_proj", "o_proj")
):
    d_model = model_config.hidden_size
    d_ff = model_config.intermediate_size
    num_layers = model_config.num_hidden_layers

    c_l = []

    for _ in range(num_layers):
        layer_cost = 0

        for module in lora_modules:
            if module in ["q_proj", "k_proj", "v_proj", "o_proj"]:
                d_in = d_out = d_model
            else:  # MLP projections if any, I assume it will have but still
                d_in = d_model
                d_out = d_ff

            layer_cost += 2 * lora_rank * (d_in + d_out)

        c_l.append(layer_cost)

    return np.array(c_l)


if __name__ == "__main__":

    # Load per-layer LoRA costs --> You get this from that json file, I assume it gives values
    flops_path = "/data/mdl-layerIF/mdl" 
    c, B = load_lora_costs(flops_path)
    print("Per-layer base costs c_l:", c)

    # Capacity ratio (MDL knob) -- Just a hyperparameter
    rho = 0.25        # try 0.1, 0.25, 0.5

    # Budget --> This will give us budget
    # B = rho * np.sum(c)
    print("Total budget B:", B)
