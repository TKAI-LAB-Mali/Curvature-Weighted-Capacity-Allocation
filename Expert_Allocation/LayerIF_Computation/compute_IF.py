from tqdm import tqdm
import pickle as pkl
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers.utils.logging import disable_progress_bar as disable_tf_progress_bar
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
from datasets import disable_progress_bar
# sys.path.append('/nas02/Hadi/Incontenxt-influence/DataInf/src')
# sys.path.insert(1, '/nas02/Hadi/Incontenxt-influence/icl-coverage/src')

from lora_model_alpha import LORAEngineGeneration
from influence import IFEngineGeneration

disable_progress_bar()
disable_tf_progress_bar()

def extract_layer_number(s):
    match = re.search(r'\.(\d+)\.', s)
    return match.group(0) if match else None

#'mrpc','cola' 'rte','commonq',

# name_list=['openbook', 'cola', 'mrpc', 'commonq', 'text_science_q_rebuttal']
# name_list = ['cola', 'mrpc', 'commonq', 'text_science_q_rebuttal']
name_list = ['text_science_q_rebuttal']

# for name in tqdm(name_list):
for name in name_list:
    print(f"======== Starting dataset: {name} ========")
    base_path = "google/gemma-7b"#"Qwen/Qwen2.5-32B"    #"mistralai/Mistral-7B-v0.1"     #"mistralai/Mistral-7B-v0.1"       #"meta-llama/Llama-2-13b-chat-hf"   
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
    # print(f"Extracted layers = {layers}\n Total layers = {len(layers)}")
    # exit()
    
    # Delete model and free GPU memory immediately after extracting layer numbers
    del base_model
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    gc.collect()
    

    print(f"Model deleted. GPU memory allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    print(f"GPU memory reserved: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
    grad_output_path = os.path.join(project_path, 'outputs', name)
    

    # Set the umask to 0 so we can control permissions fully
    current_umask = os.umask(0)

    try:
        # Create dir with 770 (rwx for owner, rwx for group, nothing for others)
        os.makedirs(grad_output_path, mode=0o770, exist_ok=True)
    finally:
        # Always restore the umask to the previous state
        os.umask(current_umask)
    # os.makedirs(grad_output_path,
    #             exist_ok=True)
    # for layer in tqdm(layers):
    for layer in layers:
        
        # print(f"layer = {layer}")
        layer_name = layer.split('.')[1] # the layers are named like .0., .1., etc, this is to retrieve the number only
        print(f"======== Processing layer {layer} ========")
        print('creating datasets')
        tokenized_datasets, collate_fn = lora_engine.create_tokenized_datasets(name)
        print("Completed creating datasets")
        tr_grad_dict, val_grad_dict = lora_engine.compute_gradient(tokenized_datasets, collate_fn, layer)

        tr_grad_dict_path = os.path.join(grad_output_path, f"training_grad_dict_{name}_{layer_name}.pkl")
        val_grad_dict_path = os.path.join(grad_output_path, f"val_grad_dict_{name}_{layer_name}.pkl")

        # with open(tr_grad_dict_path,'wb') as file:
        #     pkl.dump(tr_grad_dict, file)
        # with open(val_grad_dict_path,'wb') as file:
        #     pkl.dump(val_grad_dict, file)
        
        

        
            
        print('computing influences')
        influence_engine = IFEngineGeneration()
        influence_engine.preprocess_gradients(tr_grad_dict, val_grad_dict)
        influence_engine.compute_hvps()
        influence_engine.compute_IF()
        print(influence_engine.IF_dict['proposed'].shape)
        influence_engine.save_result(name,layer_name, model_name=base_path.split('/')[-1])
        # break
        # 1. Explicitly delete the massive objects from this iteration
        del tr_grad_dict, val_grad_dict, influence_engine
        
        # 2. Force Python to release CPU RAM immediately
        gc.collect()
    
    # del base_model
    torch.cuda.empty_cache()
    gc.collect()
