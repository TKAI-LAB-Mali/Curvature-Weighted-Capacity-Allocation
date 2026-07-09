#!/usr/bin/env python3
"""
Generate mola_lora_summary.json from a trained MoLA model or from config parameters.
This computes per-layer FLOPs for LoRA adapters.
"""

import json
import argparse
from collections import defaultdict
import torch
import os



def profile_and_save_summary(model, output_dir):
    """
    Generates mola_lora_summary.json by inspecting the model's LoRA layers.
    (Fixed to find layers by attribute existence rather than name string)
    """
    summary = {"per_layer": {}, "total_base": 0, "total_lora": 0, "total_lora_scaled": 0}
    
    # Iterate through all modules
    for name, module in model.named_modules():
        
        # 1. Check if this module is a PEFT/LoRA container
        # We look for the attributes 'lora_A' and 'lora_B' which store the adapters
        if hasattr(module, "lora_A") and hasattr(module, "lora_B"):
            
            # 2. Extract Layer Index (e.g. from "base_model.model.model.layers.10.self_attn.q_proj")
            parts = name.split('.')
            layer_idx = None
            for i, part in enumerate(parts):
                if part == "layers" and i + 1 < len(parts):
                    if parts[i+1].isdigit():
                        layer_idx = int(parts[i+1])
                        break
            
            # Fallback for models that might not use "layers" (rare but possible)
            if layer_idx is None:
                continue

            # Initialize list for this layer if new
            if str(layer_idx) not in summary["per_layer"]:
                summary["per_layer"][str(layer_idx)] = []

            # 3. Iterate through the adapters in this layer (e.g., "default_0", "default_1")
            # lora_A is typically a ModuleDict or dict mapping adapter_name -> Linear
            adapters_dict = module.lora_A
            
            for adapter_name in adapters_dict.keys():
                lora_A_layer = module.lora_A[adapter_name]
                lora_B_layer = module.lora_B[adapter_name]
                
                # Get dimensions
                r = lora_A_layer.out_features      # Rank
                in_features = lora_A_layer.in_features  # Input dim
                out_features = lora_B_layer.out_features # Output dim
                
                # Calculate Params: A (in*r) + B (out*r)
                params = (in_features * r) + (out_features * r)
                
                # FLOPs per token: 2 * params (Forward pass multiply-add)
                flops = 2 * params 

                # Create the record
                record = {
                    "module": name,
                    "adapter": adapter_name,
                    "in_features": in_features,
                    "out_features": out_features,
                    "rank": r,
                    "A_shape": [r, in_features],
                    "B_shape": [out_features, r],
                    'lora_A_weight': f"{name}.lora_A.{adapter_name}.weight",
                    'lora_B_weight': f"{name}.lora_B.{adapter_name}.weight",
                    "base_flops_per_token": 2 * in_features * out_features, 
                    "lora_flops_per_token": flops,
                    "lora_flops_per_token_scaled": flops 
                }
                
                summary["per_layer"][str(layer_idx)].append(record)
                summary["total_lora"] += flops
                summary["total_lora_scaled"] += flops
                
                # Add base model params only once per module (adapter loop shouldn't multiply base params)
                # But since we iterate adapters, we just calculate base total separately or lazily.
                # For simplicity here: we perform a rough estimation of base params total.
    
    # 4. Calculate Total Base Model FLOPs (Approximation)
    # We iterate again to avoid counting the base layer multiple times 
    # (once for every adapter attached to it)
    total_base_flops = 0
    for name, module in model.named_modules():
        if hasattr(module, "lora_A"):
             # We assume if it has lora_A, it's a Linear layer wrapper.
             # We try to get dimensions from the first adapter found, 
             # because accessing module.weight might be tricky depending on quantization/wrapping.
             try:
                 first_adapter_name = next(iter(module.lora_A.keys()))
                 first_adapter = module.lora_A[first_adapter_name]
                 
                 in_dim = first_adapter.in_features
                 # For the Output dim, we look at B because A projects to Rank
                 out_dim = module.lora_B[first_adapter_name].out_features
                 
                 total_base_flops += 2 * (in_dim * out_dim)
             except (StopIteration, KeyError, AttributeError):
                 continue

    summary["total_base"] = total_base_flops

    # Save to file
    file_path = os.path.join(output_dir, "mola_lora_summary.json")
    print(f"Saving profiling stats to {file_path}")
    with open(file_path, 'w') as f:
        json.dump(summary, f, indent=2)

    return summary


if __name__ == "__main__":
   pass
