import torch
import json
import os
import argparse
from transformers import AutoModelForCausalLM, AutoConfig

def profile_fresh_model(model_name_or_path, output_dir=".", rank=8):
    """
    Profiles a raw Hugging Face model (no LoRA).
    """
    print(f"Loading model config from: {model_name_or_path}...")
    
    # We load with device_map="meta" to avoid using RAM/GPU. 
    # We only need the structure (meta device), not the actual weights values, 
    # to calculate parameters and FLOPs.
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path, 
            device_map="meta", 
            trust_remote_code=True
        )
    except Exception as e:
        print(f"Meta load failed (older transformers version?), loading standard: {e}")
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path, 
            device_map="auto", 
            trust_remote_code=True
        )

    summary = {
        "model_name": model_name_or_path,
        "per_layer": {}, 
        "total_base": 0, 
        "total_lora": 0, 
        "total_lora_scaled": 0
    }

    print("Profiling layers...")
    
    # Iterate through all modules
    for name, module in model.named_modules():
        
        # Check if it is a Linear Layer (Standard Dense Layer)
        if isinstance(module, torch.nn.Linear):
            
            # --- 1. Filter: Only look at the Transformer Blocks ---
            # We want to match the structure of your LoRA profile, 
            # so we only care about layers inside "model.layers.X"
            # This excludes the "lm_head" and "embed_tokens".
            
            if "layers" not in name:
                continue

            # --- 2. Extract Layer Index ---
            parts = name.split('.')
            layer_idx = None
            for i, part in enumerate(parts):
                if part == "layers" and i + 1 < len(parts):
                    if parts[i+1].isdigit():
                        layer_idx = int(parts[i+1])
                        break
            
            if layer_idx is None:
                continue
            
            # Initialize list for this layer
            str_idx = str(layer_idx)
            if str_idx not in summary["per_layer"]:
                summary["per_layer"][str_idx] = []

            # --- 3. Calculate Stats ---
            in_features = module.in_features
            out_features = module.out_features
            
            # Param count for standard linear layer = in * out
            # (We usually ignore bias in FLOPs/Param counts for LLMs as it's often fused or None)
            base_params = in_features * out_features
            
            # Base FLOPs = 2 * params
            base_flops = 2 * base_params

            LoRA_flops = 2 * rank * (in_features + out_features)

            # --- 4. Create Record ---
            # We leave LoRA fields as 0 or N/A to maintain JSON compatibility
            record = {
                "module": name,
                "adapter": "base", # Placeholder
                "in_features": in_features,
                "out_features": out_features,
                "rank": rank,
                "A_shape": None,
                "B_shape": None,
                "lora_A_weight": None,
                "lora_B_weight": None,
                "base_flops_per_token": base_flops,
                "lora_flops_per_token": 0,
                "lora_flops_per_token_scaled": LoRA_flops
            }
            
            summary["per_layer"][str_idx].append(record)
            
            # Accumulate total base params/flops
            summary["total_base"] += base_flops 
            # Note: total_base in your previous logic specifically meant FLOPs, 
            # even though the variable name implies params. Keeping consistent with FLOPs here.

    # Save to file
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, "model_summary.json")
    
    print(f"Total Base FLOPs (Approximated): {summary['total_base']:,}")
    print(f"Saving statistics to {file_path}")
    
    with open(file_path, 'w') as f:
        json.dump(summary, f, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="mistralai/Mistral-7B-v0.1", help="HF Model ID")
    parser.add_argument("--output_dir", type=str, default=".", help="Where to save json")
    args = parser.parse_args()

    profile_fresh_model(args.model, args.output_dir)