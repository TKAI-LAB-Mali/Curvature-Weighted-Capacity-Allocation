"""
Implementation of Alg-1 from overleaf
"""


import argparse
import json
import os
import random
import numpy as np
import torch
import math
import util
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
# from kronfluence.task import CausalLanguageModelingTask
# from kronfluence.analyzer import Analyzer
from peft import get_peft_model, LoraConfig 

# from lib.eval import eval_ppl, eval_zero_shot
# from lib.esd_utils import get_esd_metrics

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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








def get_experts_and_cost_vector(lam, q, c, alpha, gamma, beta):
    denom = alpha + (lam * c)
    # Safety clamp for plotting/returning
    denom = np.maximum(denom, 1e-9) 
    raw_e = (gamma * (q ** beta)) / denom
    return np.maximum(raw_e, 1.0)


        

def main():

    import util_mali as mutil
    from pprint import pprint
    from get_B import load_lora_costs
    from pathlib import Path
    current_dir = Path(__file__).parent.resolve()
    Overall_project_dir = current_dir.parent
    print(f"Current directory: {current_dir}")
    print(f"Overall project directory: {Overall_project_dir}")
    influence_path = Overall_project_dir / "Expert_Allocation" / "LayerIF_Computation" / "outputs" / "layerIF_values" / "gemma-7b"
    file_attachment = "negative_IF_score_beta_3"
    file_attachment = "All_IF_score_alpha_0_5_beta_4_gamma_0.9_submitted"
    datasets = ['mrpc', 'cola', 'openbook', 'text_science_q_rebuttal','commonq']
    # datasets = ['text_science_q_rebuttal']
    # datasets = []

    print(f"Datasets to process: {datasets}")
    filed_Experts = ''
    printed_Experts = ''
    for dataset in datasets:

    
        positive = False
        layerIFs, lengths = mutil.get_IF(influence_path, dataset, 'all')

        print(f"Dataset: {dataset}, Lengths (Positive, Negative) per layer:")
        pprint(lengths)
        budget = 160 
        
        # c, B = load_lora_costs("/data/mdl-layerIF/mdl")
        c, B = load_lora_costs(current_dir)
        print(f"Original costs per expert (c_l): {c}")
        B =  0.02*B
        e_l, lambda_avg = mutil.solve_expert_allocation(np.array(layerIFs),
                                                # np.array([1.0]*len(layerIFs)),
                                                c, 
                                                np.array([0.5]*len(layerIFs)),
                                                B=B,
                                                gamma=0.9,
                                                beta=5,
                                                epsilon=0.1,
                                                max_iter=1000)
        
        e_l = np.floor(np.array(e_l))
        top_k = 2
        top_k_array = np.clip(e_l, 1, top_k).astype(int)

        e_L_times_c = e_l * c
        # print(f"Dataset: {dataset}: e_L_times_c = {np.sum(e_L_times_c)} (Budget: {B})\n")
        # print(f"e_L_times_c per layer: {e_L_times_c}\n")
        e_l_sum = np.sum(e_l)
        e_l = e_l.astype(int).tolist()


        
        e_l = str(e_l).replace('[','').replace(']','').replace(' ', '')
        top_k_list = str(top_k_array.tolist()).replace('[','').replace(']','').replace(' ', '')

        bash_case = f'  *"{dataset}"*)\n  current_experts="{e_l}"\n  current_top_k="{top_k_list}"\n  ;; \n'
        scienceq_case = f'  *"scienceq"*)\n  current_experts="{e_l}"\n  current_top_k="{top_k_list}"\n  ;; \n'
        
        filed_Experts += scienceq_case if dataset == 'text_science_q_rebuttal' else bash_case
        printed_Experts += f"{dataset}: {e_l} : Total Experts = {e_l_sum}\n" 

    

        # print(f"MDL Expert Allocation Results old: {np.array(lambda_avg_old)}, {np.floor(e_l_old)}")
        # print(f"MDL Expert Allocation Results new: {e_l}")
    print("Final Expert Allocations per dataset:")
    print(printed_Experts)
    
    with open(f'MDL_allocated_experts_gemma_{file_attachment}.txt', 'w') as f:
        f.write(filed_Experts)
    



if __name__ == '__main__':
    main()

    