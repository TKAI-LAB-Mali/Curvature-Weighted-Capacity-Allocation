from transformers import AutoConfig, AutoModelForCausalLM
import torch
import torch.nn as nn
import json
import os
import pickle as pkl
import numpy as np

def get_IF(experts_path, dataset):
    # path = '/data/mdl-layerIF/Expert_Allocation/layerIF_Computation/outputs/layerIF_values/mistral-7B-v0.1'
    layer_IFs = []
    for file in sorted(os.listdir(experts_path)):
        if file.endswith(dataset + '.pkl'):
            results = pkl.load(open(os.path.join(experts_path, file), 'rb'))
            layerIF_value = results['influence']['proposed'].to_numpy()
            layer_IFs.append(-1.0*np.sum(layerIF_value))
    return layer_IFs

    import json

def find_layers(module:nn.Module, layers=[nn.Linear], name=''):
    """
    Recursively find the layers of a certain type in a module.

    Args:
        module (nn.Module): PyTorch module.
        layers (list): List of layer types to find.
        name (str): Name of the module.

    Returns:
        dict: Dictionary of layers of the given type(s) within the module.
    """
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(
            child, layers=layers, name=name + '.' + name1 if name != '' else name1
        ))
    return res

def get_layer_param_count(model, device=torch.device('cuda:0')):
    """
    Based on pruning code for magnitude/wanda/sparsegpt in LayerIF, get number of parameters available for sparsification.
    
    :param model: neural network model
    :param device: cpu/gpu
    
    Return:
    layerwise_param_count (list): parameter count per layer that will be used for sparsification
    """
    if 'OPT' in model.__class__.__name__:
        layers = model.model.decoder.layers
    else:
        layers = model.model.layers

    layer_num = len(find_layers(layers))
    layerwise_param_count = []

    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)
        count = 0
        for name in subset:
            W = subset[name].weight.data
            # print(f"layer: {i}, name: {name}, elements: {W.numel()}")
            count += W.numel()
        layerwise_param_count.append(count)
    
    return layerwise_param_count


def get_all_layer_connections(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)

    # print(f"{'Layer':<10} | {'Base Connections':<20}")
    # print("-" * 35)

    total_model_connections = 0
    
    # Iterate through keys '0' through '31' numerically
    # The JSON keys are strings, so we sort them as integers
    layer_indices = sorted(data.get('per_layer', {}).keys(), key=lambda x: int(x))

    per_layer_connections = []
    for layer_idx in layer_indices:
        layer_items = data['per_layer'][layer_idx]
        
        unique_modules = set()
        layer_count = 0
        
        for item in layer_items:
            module_name = item['module']
            
            # Only count each base module once per layer
            # (Your JSON repeats the module name for each expert adapter attached to it)
            if module_name not in unique_modules:
                unique_modules.add(module_name)
                
                # Calculate connections: rows * columns
                connections = item['in_features'] * item['out_features']
                layer_count += connections
        per_layer_connections.append(layer_count)
        total_model_connections += layer_count
        # print(f"Layer {layer_idx:<4} | {layer_count:,}")

    # print("-" * 35)
    # print(f"TOTAL:      {total_model_connections:,}")

    return per_layer_connections
# Run
# get_all_layer_connections('mola_lora_summary.json')