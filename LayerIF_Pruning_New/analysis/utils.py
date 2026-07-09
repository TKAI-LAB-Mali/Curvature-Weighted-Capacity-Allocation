import os
import json
import torch
import numpy as np


# ---------------------------------------------------------------------------
# Weight norms
# ---------------------------------------------------------------------------

def get_weight_norms(model, layer_prefix="model.layers"):
    """
    Per-submodule L2 norms, grouped by transformer layer index.
    Returns: {layer_idx: {submodule_name: norm_value}}
    
    Example key: {5: {"self_attn.q_proj": 42.3, "mlp.gate_proj": 18.1, ...}}
    """
    weight_norms = {}
    with torch.no_grad():
        for name, param in model.named_parameters():
            if layer_prefix not in name or "weight" not in name:
                continue

            # --- NEW: Explicitly ignore all LayerNorm parameters ---
            if "layernorm" in name.lower():
                continue
            parts = name.split(".")
            try:
                layer_idx = int(parts[2])
            except (ValueError, IndexError):
                continue

            # Robust submodule key: strip prefix and ".weight" suffix
            sub_module = (
                name
                .removeprefix(f"{layer_prefix}.{layer_idx}.")
                .removesuffix(".weight")
            )
            norm_value = torch.norm(param.data.float(), p=2).item()

            if layer_idx not in weight_norms:
                weight_norms[layer_idx] = {}
            weight_norms[layer_idx][sub_module] = norm_value

    return weight_norms


def aggregate_weight_norms_from_dict(weight_norms_dict):
    """
    Collapse a {layer: {submodule: norm}} dict to {layer: aggregated_norm} using
    sum-of-squares aggregation (true L2 norm of the full layer parameter vector).

    Accepts string or int layer keys (as loaded from JSON).
    This is the single source of truth for aggregation — usable both at
    runtime and at plot time without needing a live model.
    """
    return {
        int(layer): float(np.sqrt(sum(v ** 2 for v in submodules.values())))
        for layer, submodules in weight_norms_dict.items()
    }


def get_aggregated_weight_norms(model, layer_prefix="model.layers"):
    """
    True L2 norm of all weights in each transformer layer.
    Delegates to aggregate_weight_norms_from_dict so aggregation
    logic lives in exactly one place.
    Returns: {layer_idx: float}
    """
    return aggregate_weight_norms_from_dict(get_weight_norms(model, layer_prefix))


# ---------------------------------------------------------------------------
# Activation norms
# ---------------------------------------------------------------------------

def get_activation_norms(model, tokenizer, device, n_samples=4, seq_len=128,
                         layer_prefix="model.layers"):
    """
    Run a short forward pass and record the L2 norm of each transformer
    layer's output tensor.

    Uses a fixed prompt so results are reproducible and comparable across
    before/after pruning calls.

    Returns: {layer_idx: float}
    """
    # Define a pool of diverse contexts (Language, Code, QA, Math/Logic)
    prompt_pool = [
        "The quick brown fox jumps over the lazy dog because it needs to test the model's text processing capabilities.",
        "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)",
        "User: Explain the theory of relativity.\nAssistant: The theory of relativity, developed by Albert Einstein, states that",
        "Solve for x: 2x + 4 = 12. First, subtract 4 from both sides to get 2x = 8. Then, divide by 2"
    ]
    
    # Tile the prompts if n_samples > len(prompt_pool)
    prompts = [prompt_pool[i % len(prompt_pool)] for i in range(n_samples)]

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        truncation=True,
        max_length=seq_len,
        padding=True,
    )
    # Move input ids to the first device; device_map handles the rest
    input_ids = inputs["input_ids"].to(device)

    activation_norms = {}
    hooks = []

    # Identify transformer layer modules by name depth
    target_depth = layer_prefix.count(".") + 1
    for name, module in model.named_modules():
        if not name.startswith(layer_prefix):
            continue
        if name.count(".") != target_depth:
            continue
        try:
            layer_idx = int(name.split(".")[-1])
        except ValueError:
            continue

        def make_hook(idx):
            def hook(module, inp, output):
                out = output[0] if isinstance(output, tuple) else output
                # Move to CPU immediately to avoid OOM on large models
                activation_norms[idx] = torch.norm(out.detach().float().cpu(), p=2).item()
            return hook

        hooks.append(module.register_forward_hook(make_hook(layer_idx)))

    # with torch.no_grad():
    #     model(input_ids)

    # for h in hooks:
    #     h.remove()

    try:
        with torch.no_grad():
            model(input_ids)
    finally:
        for h in hooks:
            h.remove()

    return activation_norms


# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------

def save_norms(weight_norms, activation_norms, save_dir, tag):
    """
    Persist per-submodule weight norms and per-layer activation norms.

    Files created:
        {save_dir}/weight_norms_{tag}.json   — {layer: {submodule: norm}}
        {save_dir}/activation_norms_{tag}.json — {layer: norm}
    """
    os.makedirs(save_dir, exist_ok=True)
    # JSON requires string keys
    with open(os.path.join(save_dir, f"weight_norms_{tag}.json"), "w") as f:
        json.dump({str(k): v for k, v in sorted(weight_norms.items())}, f, indent=2)
    with open(os.path.join(save_dir, f"activation_norms_{tag}.json"), "w") as f:
        json.dump({str(k): float(v) for k, v in sorted(activation_norms.items())}, f, indent=2)


def load_norms(save_dir, tag):
    """
    Load previously saved norm files. Returns (weight_norms, activation_norms)
    with integer layer keys.
    """
    with open(os.path.join(save_dir, f"weight_norms_{tag}.json")) as f:
        raw_w = json.load(f)
    with open(os.path.join(save_dir, f"activation_norms_{tag}.json")) as f:
        raw_a = json.load(f)

    weight_norms = {int(k): v for k, v in raw_w.items()}
    activation_norms = {int(k): float(v) for k, v in raw_a.items()}
    return weight_norms, activation_norms