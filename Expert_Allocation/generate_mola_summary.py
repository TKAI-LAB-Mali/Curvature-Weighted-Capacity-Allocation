#!/usr/bin/env python3
"""
Generate mola_lora_summary.json from a trained MoLA model or from config parameters.
This computes per-layer FLOPs for LoRA adapters.
"""

import json
import argparse
from collections import defaultdict

# Model architectures
MODEL_CONFIGS = {
    "mistral": {
        "hidden_size": 4096,
        "intermediate_size": 14336,
        "num_hidden_layers": 32,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,  # GQA
    },
    "gemma": {
        "hidden_size": 3072,
        "intermediate_size": 24576,
        "num_hidden_layers": 28,
        "num_attention_heads": 16,
        "num_key_value_heads": 16,  # No GQA (MHA)
    },
}

def compute_lora_flops(in_features, out_features, rank):
    """
    LoRA FLOPs per token = 2 * (in_features * rank + rank * out_features)
    Simplified: 2 * rank * (in_features + out_features)
    """
    return 2 * rank * (in_features + out_features)

def compute_base_flops(in_features, out_features):
    """Base linear layer FLOPs per token = 2 * in_features * out_features"""
    return 2 * in_features * out_features

def generate_mola_summary(
    number_experts: list,
    lora_r: list,
    model_type: str = "mistral",
    target_modules: list = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "down_proj", "up_proj"],
    output_path: str = "mola_lora_summary.json"
):
    """
    Generate mola_lora_summary.json
    
    Args:
        number_experts: List of number of experts per layer (length = num_layers)
        lora_r: List of LoRA ranks per layer (length = num_layers)
        model_type: "mistral" or "gemma"
        target_modules: List of module names to apply LoRA
        output_path: Output file path
    """
    if model_type not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model_type: {model_type}. Supported: {list(MODEL_CONFIGS.keys())}")
    
    config = MODEL_CONFIGS[model_type]
    hidden_size = config["hidden_size"]
    intermediate_size = config["intermediate_size"]
    num_layers = config["num_hidden_layers"]
    num_kv_heads = config["num_key_value_heads"]
    head_dim = hidden_size // config["num_attention_heads"]
    kv_size = num_kv_heads * head_dim
    
    print(f"Model: {model_type}")
    print(f"  hidden_size: {hidden_size}, intermediate_size: {intermediate_size}")
    print(f"  num_layers: {num_layers}, kv_size: {kv_size}")
    
    # Validate number_experts length
    if len(number_experts) != num_layers:
        raise ValueError(f"number_experts has {len(number_experts)} values but model has {num_layers} layers")
    
    # Module dimensions
    module_dims = {
        "q_proj": (hidden_size, hidden_size),      
        "k_proj": (hidden_size, kv_size),          
        "v_proj": (hidden_size, kv_size),          
        "o_proj": (hidden_size, hidden_size),      
        "gate_proj": (hidden_size, intermediate_size),
        "up_proj": (hidden_size, intermediate_size),
        "down_proj": (intermediate_size, hidden_size),
    }
    
    per_layer = defaultdict(list)
    total_base = 0
    total_lora = 0
    total_lora_scaled = 0
    
    for layer_idx in range(num_layers):
        n_experts = number_experts[layer_idx]
        rank = lora_r[layer_idx]
        
        for module_name in target_modules:
            if module_name not in module_dims:
                continue
                
            in_features, out_features = module_dims[module_name]
            base_flops = compute_base_flops(in_features, out_features)
            lora_flops = compute_lora_flops(in_features, out_features, rank)
            
            # Add entry for each expert
            for expert_idx in range(n_experts):
                entry = {
                    "module": f"base_model.model.model.layers.{layer_idx}.self_attn.{module_name}" 
                             if module_name in ["q_proj", "k_proj", "v_proj", "o_proj"]
                             else f"base_model.model.model.layers.{layer_idx}.mlp.{module_name}",
                    "adapter": f"default_{expert_idx}",
                    "in_features": in_features,
                    "out_features": out_features,
                    "rank": rank,
                    "A_shape": [rank, in_features],
                    "B_shape": [out_features, rank],
                    "lora_A_weight": f"base_model.model.model.layers.{layer_idx}.{'self_attn' if module_name in ['q_proj','k_proj','v_proj','o_proj'] else 'mlp'}.{module_name}.lora_A.default_{expert_idx}.weight",
                    "lora_B_weight": f"base_model.model.model.layers.{layer_idx}.{'self_attn' if module_name in ['q_proj','k_proj','v_proj','o_proj'] else 'mlp'}.{module_name}.lora_B.default_{expert_idx}.weight",
                    "base_flops_per_token": base_flops,
                    "lora_flops_per_token": lora_flops,
                    "lora_flops_per_token_scaled": lora_flops,
                }
                per_layer[str(layer_idx)].append(entry)
                total_lora += lora_flops
                total_lora_scaled += lora_flops
            
            # Base FLOPs counted once per module (not per expert)
            total_base += base_flops
    
    summary = {
        "per_layer": dict(per_layer),
        "total_base": total_base,
        "total_lora": total_lora,
        "total_lora_scaled": total_lora_scaled,
    }
    
    with open(output_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"Generated {output_path}")
    print(f"  Total base FLOPs: {total_base:,}")
    print(f"  Total LoRA FLOPs: {total_lora:,}")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate mola_lora_summary.json")
    parser.add_argument("--model", type=str, default="mistral", choices=["mistral", "gemma"],
                        help="Model type: mistral or gemma")
    parser.add_argument("--number_experts", type=str, required=True,
                        help="Comma-separated list of experts per layer")
    parser.add_argument("--lora_r", type=str, default="8",
                        help="Comma-separated LoRA ranks per layer (or single value)")
    parser.add_argument("--output", type=str, default="mola_lora_summary.json",
                        help="Output file path")
    
    args = parser.parse_args()
    
    # Parse number_experts
    number_experts = [int(x) for x in args.number_experts.split(",")]
    
    # Get number of layers for this model
    num_layers = MODEL_CONFIGS[args.model]["num_hidden_layers"]
    
    # Parse lora_r (can be single value or per-layer)
    lora_r_values = [int(x) for x in args.lora_r.split(",")]
    if len(lora_r_values) == 1:
        lora_r = lora_r_values * num_layers
    else:
        lora_r = lora_r_values
    
    generate_mola_summary(number_experts, lora_r, model_type=args.model, output_path=args.output)