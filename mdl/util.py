from transformers import AutoConfig, AutoModelForCausalLM
import torch

import json
import os
import pickle as pkl
import numpy as np

def get_IF(experts_path, dataset, positive: bool):
    # path = '/data/mdl-layerIF/Expert_Allocation/layerIF_Computation/outputs/layerIF_values/mistral-7B-v0.1'
    layer_IFs = []
    for file in sorted(os.listdir(experts_path)):
        if file.endswith(dataset + '.pkl'):
            results = pkl.load(open(os.path.join(experts_path, file), 'rb'))
            layerIF_value = results['influence']['proposed'].to_numpy()

            if positive:
                summed_vector = np.sum(layerIF_value, axis=0)
                remaining_values = summed_vector[summed_vector>0]
                if remaining_values.size == 0:
                    remaining_values = np.array([0])
                layer_IFs.append(np.sum(remaining_values))
            else:
                layer_IFs.append(np.sum(layerIF_value))

    layer_IFs = np.array(layer_IFs)
    # print(f"Layer IFs before inversion: {layer_IFs}")

    inverted_layer_IFs = np.max(layer_IFs) - layer_IFs
    return inverted_layer_IFs #+ 1e-6  # to avoid zero values






def get_all_layer_connections(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)

    print(f"{'Layer':<10} | {'Base Connections':<20}")
    print("-" * 35)

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
        print(f"Layer {layer_idx:<4} | {layer_count:,}")

    print("-" * 35)
    print(f"TOTAL:      {total_model_connections:,}")

    return per_layer_connections
# Run
# get_all_layer_connections('mola_lora_summary.json')