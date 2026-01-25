from dataloader import create_dataloaders, load_noisy_dataset_by_task
from lora_model_alpha import LORAEngine
from influence import IFEngine

from tqdm import tqdm
import pickle as pkl
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForSequenceClassification,
    
    
    
    
    get_linear_schedule_with_warmup,
    BitsAndBytesConfig,
    LlamaForCausalLM,
    LlamaTokenizer,
    AutoModelForCausalLM
)
import re

import numpy as np
import os


def extract_layer_number(s):
    match = re.search(r'\.(\d+)\.', s)
    return match.group(0) if match else None


def get_model_layers():
    base_path = "google/gemma-7b"     #"mistralai/Mistral-7B-v0.1"    #"google/gemma-7b"    

    base_model = AutoModelForCausalLM.from_pretrained(
        base_path,
        #quantization_config=quantization_config,
        load_in_4bit=True,
        torch_dtype=torch.bfloat16,
        offload_folder="offload",
        offload_state_dict=True,
        device_map='auto'
    )
    layers={}
    for k,v in base_model.named_parameters():
        #print(k)
        result = extract_layer_number(k)
        layers[result]=0

    layers=list(layers.keys())

    layers.remove(None)
    return layers

def scale_values_final(values, target_sum, exponent=1.5):
    values = np.array(values, dtype=np.float64)
    n = len(values)

    

    
    inverted_values = np.max(values) - values
    if np.all(inverted_values == 0):
        inverted_values += 1e-6

    scaled_values = np.power(inverted_values, exponent)
    scaled_fractions = (scaled_values / scaled_values.sum()) * (target_sum - n)  # Subtract 1 per value

    # Floor and add the minimum value of 1
    floored = np.floor(scaled_fractions).astype(int) + 1
    remainder = target_sum - floored.sum()

    # Distribute remaining units to values with largest fractional parts
    fractional_parts = scaled_fractions - np.floor(scaled_fractions)
    top_indices = np.argsort(fractional_parts)[::-1]

    for i in range(remainder):
        floored[top_indices[i]] += 1

    return floored


#All IF values 


def expert_allocator(names=['mrpc', 'cola', 'openbook', 'text_scienceq','commonq'],
                     layer_IF_path='', experts_paths=''):
    
    # layers = get_model_layers()
    layers = ['.0', '.1', '.2', '.3', '.4', '.5', '.6', '.7', '.8', '.9', '.10', '.11', '.12', '.13', '.14', '.15', '.16', '.17', '.18', '.19', '.20', '.21', '.22', '.23', '.24', '.25', '.26', '.27']



    base_dir = os.path.dirname(os.path.abspath(__file__))
    if experts_paths == '':
        experts_path = os.path.join(base_dir, 'allocated_experts.txt')
    else:
        experts_path = experts_paths

    if layer_IF_path == '':        
        layer_IF_path = os.path.join(base_dir, 'outputs', 'layerIF_values', 'gemma-7b')

    open(experts_path, 'w').close()  # Clear the file before writing 

    for name in names:
        IFs = []
        for layer in layers:
            layer_number = layer.split('.')[1] # the layers are named like .0., .1., etc, this is to retrieve the number only
            path = os.path.join(layer_IF_path, f"results_gemma-7b_{layer_number}_{name}.pkl")

            with open(path, 'rb') as f:
                IF_pickle=pkl.load(f)
            IF_pickle_new=IF_pickle['influence']['proposed'].to_numpy()
            # print(IF_pickle_new)
            summed_vector = np.sum(IF_pickle_new, axis=0, keepdims=True)
            print(f'summed_vector shape for layer {layer_number}: {summed_vector.shape}')
        
            
            IFs.append(np.sum(summed_vector))
        
        scaled_numbers_fixed = scale_values_final(IFs, target_sum=160, exponent=2) #Can change target sum and exponent here
        print(name)
        # print(list(scaled_numbers_fixed))
        # print(scaled_numbers_fixed.tolist())
        scaled_number_str = str(scaled_numbers_fixed.tolist()).replace('[','').replace(']','').replace(' ', '')
        print(scaled_number_str)
        with open(experts_path, 'a') as f:
            f.write(f"{name}: {scaled_number_str}\n")

#Only Positive IF values

def Our_positive_IF_(IF_pickle_new):
    summed_vector = np.sum(IF_pickle_new, axis=0) 
    remaining_values = summed_vector[summed_vector>0]
    if remaining_values.size == 0:
        return 0
    else:
        return np.sum(remaining_values)
    
def Hadi_positive_IF_(IF_pickle_new):
    summed_vector = np.sum(IF_pickle_new, axis=0, keepdims=True)
    remaining_values = np.delete(summed_vector, np.where(summed_vector > 0)[1])
    return np.sum(remaining_values)

    

def expert_allocator_positive_IF(names=['mrpc', 'cola', 'openbook', 'text_scienceq','commonq'],
                     layer_IF_path='', experts_paths=''):
    
    layers = get_model_layers()


    base_dir = os.path.dirname(os.path.abspath(__file__))
    if experts_paths == '':
        experts_path = os.path.join(base_dir, 'allocated_experts_positive_IF.txt')
    else:
        experts_path = experts_paths

    if layer_IF_path == '':        
        layer_IF_path = os.path.join(base_dir, 'outputs', 'layerIF_values', 'gemma-7b')

    open(experts_path, 'w').close()  # Clear the file before writing 
    scaled_IFs = []
    Hadi_scaled_IFs = []
    for name in names:
        IFs = []
        IFs_Hadi = []
        for layer in layers:
            layer_number = layer.split('.')[1] # the layers are named like .0., .1., etc, this is to retrieve the number only
            path = os.path.join(layer_IF_path, f"results_gemma-7b_{layer_number}_{name}.pkl")

            with open(path, 'rb') as f:
                IF_pickle=pkl.load(f)
            IF_pickle_new=IF_pickle['influence']['proposed'].to_numpy()


            # summed_vector = np.sum(IF_pickle_new, axis=0) 
            # remaining_values = summed_vector[summed_vector>0]
            # print(f'Layer {layer_number}, remaining values size: {remaining_values.size}')
            # if remaining_values.size == 0:
            #     IFs.append(0)
            # else:
            #     IFs.append(np.sum(remaining_values))

            positive_IF_value = Our_positive_IF_(IF_pickle_new)
            IFs.append(positive_IF_value)

            positive_IF_value_Hadi = Hadi_positive_IF_(IF_pickle_new)
            IFs_Hadi.append(positive_IF_value_Hadi)


        print(f"Positive IF values for {name}: {IFs}")
        scaled_numbers_fixed = scale_values_final(IFs, target_sum=160, exponent=3) #Can change target sum and exponent here
        scaled_numbers_fixed_Hadi = scale_values_final(IFs_Hadi, target_sum=160, exponent=3) #Can change target sum and exponent here
        scaled_IFs.append(f"{name}: {scaled_numbers_fixed.tolist()}\n")
        Hadi_scaled_IFs.append(f"{name}: {scaled_numbers_fixed_Hadi.tolist()}\n")

    
    with open(experts_path, 'a') as f:
        for line in scaled_IFs:
            line = line.replace('[','').replace(']','').replace(' ', '')
            f.write(line)
        f.write("\nHadi Method Positive IF Allocation:\n")
        for line in Hadi_scaled_IFs:
            line = line.replace('[','').replace(']','').replace(' ', '')
            f.write(line)

#Top k ratio IF values

def expert_allocator_topk_ratio_IF(names=['mrpc', 'cola', 'openbook', 'text_scienceq','commonq'],
                     layer_IF_path='', experts_paths='',
                     top_k_ratio=0.1):
    
    layers = get_model_layers()


    base_dir = os.path.dirname(os.path.abspath(__file__))
    if experts_paths == '':
        experts_path = os.path.join(base_dir, 'allocated_experts_topk_ratio_IF.txt')
    else:
        experts_path = experts_paths

    if layer_IF_path == '':        
        layer_IF_path = os.path.join(base_dir, 'outputs', 'layerIF_values', 'gemma-7b')

    open(experts_path, 'w').close()  # Clear the file before writing 

    for name in names:
        IFs = []
        for layer in layers:
            layer_number = layer.split('.')[1] # the layers are named like .0., .1., etc, this is to retrieve the number only
            path = os.path.join(layer_IF_path, f"results_gemma-7b_{layer_number}_{name}.pkl")

            with open(path, 'rb') as f:
                IF_pickle=pkl.load(f)
            IF_pickle_new=IF_pickle['influence']['proposed'].to_numpy()
            # print(IF_pickle_new)
            summed_vector = np.sum(IF_pickle_new, axis=0)
        
            num_values_to_keep = int(len(summed_vector) * keep_ratio)
            threshold = np.partition(summed_vector, num_values_to_keep)[num_values_to_keep]
            
            # Keep only the most negative values below the threshold
            remaining_values = summed_vector[summed_vector <= threshold]
            IFs.append(np.sum(remaining_values))
        
        scaled_numbers_fixed = scale_values_final(IFs, target_sum=160, exponent=2) #Can change target sum and exponent here
        print(name)
        # print(list(scaled_numbers_fixed))
        print(scaled_numbers_fixed.tolist())
        with open(experts_path, 'a') as f:
            f.write(f"{name}: {scaled_numbers_fixed.tolist()}\n")



if __name__ == "__main__":

    # expert_allocator(names=['mrpc', 'cola', 'openbook', 'commonq', 'text_science_q_rebuttal'],
    #                  layer_IF_path='')
    
    expert_allocator_positive_IF(names=['mrpc', 'cola', 'openbook', 'text_science_q_rebuttal','commonq'],
                     layer_IF_path='')

    # expert_allocator_topk_ratio_IF(names=['mrpc', 'cola', 'openbook', 'text_science_q_rebuttal','commonq'],
    #                  layer_IF_path='',
    #                  top_k_ratio=0.25)

    
    