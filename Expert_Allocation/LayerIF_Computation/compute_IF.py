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
import argparse
import os
import gc

os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3"



import sys
# sys.path.append('/nas02/Hadi/Incontenxt-influence/DataInf/src')
# sys.path.insert(1, '/nas02/Hadi/Incontenxt-influence/icl-coverage/src')

from lora_model_alpha import LORAEngineGeneration
from influence import IFEngineGeneration

def extract_layer_number(s):
    match = re.search(r'\.(\d+)\.', s)
    return match.group(0) if match else None

#'mrpc','cola' 'rte','commonq',

name_list=['commonq', 'text_science_q_rebuttal']  #' openbook', '' , 'mrpc', 'cola', commonq

for name in tqdm(name_list):
    base_path = "mistralai/Mistral-7B-v0.1" #"Qwen/Qwen2.5-32B"    #"mistralai/Mistral-7B-v0.1"       #"google/gemma-7b" #  #"meta-llama/Llama-2-13b-chat-hf"   # 
    project_path ="/data/mdl-layerIF/Expert_Allocation/LayerIF_Computation" 
    lora_engine = LORAEngineGeneration(base_path=base_path, 
                                    project_path=project_path,
                                    dataset_name=name)

    
    bnb_cfg = BitsAndBytesConfig(
    load_in_4bit=True,                       # default
    llm_int8_enable_fp32_cpu_offload=True,  # 
     )
    
    base_model = AutoModelForCausalLM.from_pretrained(
        base_path,
        #quantization_config=bnb_cfg,
        # load_in_8bit=True,
        torch_dtype=torch.bfloat16,
        offload_folder="offload",
        offload_state_dict=True,
        device_map='auto'
    )
    
    # base_model.to('cpu')

    layers={}
    for k,v in base_model.named_parameters():
        #print(k)
        result = extract_layer_number(k)
        layers[result]=0

    layers=list(layers.keys())

    layers.remove(None)
    
    # Delete model and free GPU memory immediately after extracting layer numbers
    del base_model
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    gc.collect()
    

    print(f"Model deleted. GPU memory allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    print(f"GPU memory reserved: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")

    for layer in tqdm(layers):
        print(layer)
        print('creating datasets')
        tokenized_datasets, collate_fn = lora_engine.create_tokenized_datasets(name)
        tr_grad_dict, val_grad_dict = lora_engine.compute_gradient(tokenized_datasets, collate_fn, layer)

        # with open(f"./training_grad_dict_{name}_{layer}.pkl",'wb') as file:
        #     pkl.dump(tr_grad_dict, file)
        # with open(f"./val_grad_dict_{name}_{layer}.pkl",'wb') as file:
        #     pkl.dump(val_grad_dict, file)
            
        print('computing influences')
        influence_engine = IFEngineGeneration()
        influence_engine.preprocess_gradients(tr_grad_dict, val_grad_dict)
        influence_engine.compute_hvps()
        influence_engine.compute_IF()
        print(influence_engine.IF_dict['proposed'].shape)
        influence_engine.save_result(name,layer, model_name=base_path.split('/')[-1])

        # 1. Explicitly delete the massive objects from this iteration
        # del tr_grad_dict, val_grad_dict, influence_engine
        
        # 2. Force Python to release CPU RAM immediately
        gc.collect()
    
    # del base_model
    torch.cuda.empty_cache()
    gc.collect()
