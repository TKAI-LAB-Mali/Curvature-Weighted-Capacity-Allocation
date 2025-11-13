import argparse
import json
import os
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from lib.eval import eval_ppl, eval_zero_shot
from lib.esd_utils import get_esd_metrics


def get_llm(model, cache_dir='llm_weights'):
    model = AutoModelForCausalLM.from_pretrained(
        model,
        torch_dtype=torch.float16,
        cache_dir=cache_dir,
        low_cpu_mem_usage=True,
        device_map='auto'
    )

    model.seqlen = 2048
    return model


def get_flops(path):
    per_layer_base_flops = []
    per_layer_lora_flops_scaled = []
    with open(os.path.join(path, 'mola_lora_summary.json'), 'r') as f:
        summary = json.load(f)
    
    per_layer = summary.get('per_layer', {})
    
    for key, val in per_layer.items():
        base_flops = 0
        lora_flops = 0
        for attn_head in val:
            base_flops += attn_head['base_flops_per_token']
            lora_flops += attn_head['lora_flops_per_token_scaled']
        per_layer_base_flops.append(base_flops)
        per_layer_lora_flops_scaled.append(lora_flops)
    
    return per_layer_base_flops, per_layer_lora_flops_scaled


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, help='model type')
    parser.add_argument('--seed', type=int, default=0, help='Seed for sampling the calibration data')
    parser.add_argument('--nsamples', type=int, default=128, help='Number of calibration samples')
    parser.add_argument('--cache_dir', default='llm_weights', type=str)
    parser.add_argument('--save', type=str, default=None, help='Path to save results')
    parser.add_argument('--save_model', type=str, default=None, help='Path to save the model')
    
    
    flops_paths = "../Expert_Allocation/layerIF_outputs/mistral_mola_46810_224_glue_cola_all"

