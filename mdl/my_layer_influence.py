import os
import sys
import fire
import torch
import transformers
from transformers import Trainer
from datasets import load_dataset
from datasets import load_from_disk
import torch.nn as nn
import numpy as np
from tqdm import tqdm

# PATH SETUP
# Add the Expert_Allocation folder to sys.path so 
# imports work as if we were in that folder
current_dir = os.path.dirname(os.path.abspath(__file__))
expert_alloc_path = os.path.join(current_dir, '../Expert_Allocation')
sys.path.append(expert_alloc_path)

# now we don't need to add prefix of Expert_Allocation. to the below imports
from peft import prepare_model_for_int8_training
from src.mola_mapping_hacked import get_peft_model
from src.mola_lora_hacked import LoraConfig
from src.mola_peft_model_hacked import set_peft_model_state_dict_moe
from src.mola_modeling_mistral_hacked import MistralForCausalLM_d
from src.mistralconfig import MistralConfig
from utils.prompter import Prompter
from transformers import LlamaTokenizer, AutoConfig
from transformers import AutoTokenizer
from transformers import TrainerCallback

import wandb
import json

os.environ['WANDB_DISABLED'] = 'false'

import random
seed = 42
random.seed(seed)
torch.manual_seed(seed)

def get_layer_indices(model):
    """
    Return a dictionary mapping layer index (int) to a list of parameter indices in the flattened gradient vector
    Args:
        model (nn.Module): Neural Network model
    Return:
        layer_indices (dict[int, list]): dictionary[layer idx] -> [parameter idxs]
    """
    layer_indices = {}
    current_idx = 0
    
    for name, p in model.named_parameters():
        if p.requires_grad:
            num_params = p.numel()

            # Logic to extract layer ID from parameter name
            # Example name: "base_model.model.layers.0.self.attn.q_proj.lora_A.weight"
            parts = name.split('.')
            layer_id = -1
            for part in parts:
                if part.isdigit():
                    layer_id = int(part)
                    break
            if layer_id != -1:
                if layer_id not in layer_indices:
                    layer_indices[layer_id] = []

                # store tje range of indices for this parameter
                layer_indices[layer_id].extend(range(current_idx, current_idx + num_params))

            current_idx += num_params
    
    return layer_indices

def flatten_gradients(model):
    """
    Flattens the gradients of trainable parameters into a single vector
    Args:
        model (nn.Module): Neural Network model
    Return:
        (torch.tensor): tensor of flattened gradients for each parameter
    """
    grads = []
    for p in model.parameters():
        if p.requires_grad:
            if p.grad is not None:
                grads.append(p.grad.view(-1))
            else:
                # If a param didn't get a grad (eg. unused), treat as zero
                grads.append(torch.zeros_like(p.view(-1)))
    return torch.cat(grads)

def compute_gradients(model, batch, device):
    """
    Compute gradients for a single batch (size 1).
    Args:
        model (nn.Module): Neural Network model
        batch ():
        device (torch.device): cpu/gpu device
    Return:
        (torch.tensor): tensor of flattened gradients for all model parameters
    """
    model.zero_grad()

    # Move batch to device
    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    labels = batch['labels'].to(device)

    # Forward pass
    outputs = model(input_ids=input_ids, attention_mask=attention_mask, 
                    labels=labels)
    loss = outputs.loss

    # Backward pass
    loss.backward()

    return flatten_gradients(model)



def main(
    base_model: str = "mistralai/Mistral-7B-v0.1",
    data_path: str = "./sampled_data/sampled_scienceqa_train_all.hf",
    output_dir: str = "./layer_influence_output",
    lora_r: str = "8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8",
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    lora_target_modules: str = "q_proj,v_proj,k_proj,o_proj,gate_proj,down_proj,up_proj",
    number_experts: str = "2,2,2,2,2,2,2,2,4,4,4,4,4,4,4,4,6,6,6,6,6,6,6,6,8,8,8,8,8,8,8,8",
    top_k: str = "2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2",
):
    # --- Argument Parsing (Boilerplate) ---
    if isinstance(lora_r, str):
        lora_r = [int(x) for x in lora_r.split(",")]
        number_experts = [int(x) for x in number_experts.split(",")]
        top_k = [int(x) for x in top_k.split(",")]
        lora_target_modules = lora_target_modules.split(",")

    device_map = "auto"
    
    print(f"Loading model: {base_model}")

    # ==========================================
    # TODO: YOUR CODE HERE
    
    batch_size = 128
    micro_batch_size = 8
    obalance = False
    gradient_accumulation_steps = batch_size // micro_batch_size
    

    # 1. Initialize tokenizer (set pad_token_id=0, padding_side="left")
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    tokenizer.pad_token_id = (
        0   # unk. we want this to be different from the eos token
    )
    tokenizer.padding_side = 'left'

    # 2. Load MistralConfig
    config = MistralConfig.from_pretrained(base_model)
    config.lora_target_modules = lora_target_modules
    
    # 3. Load MistralForCausalLM_d
    model = MistralForCausalLM_d.from_pretrained(
        base_model,
        config=config,
        load_in_8bit=False,
        torch_dtype=torch.float16,
        device_map=device_map,
        attn_implementation='eager'
    )
    model.get_new_parameters(number_experts, top_k, obalance)
    
    # 4. Create LoraConfig
    config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=lora_target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        number_experts=number_experts,
        top_k=top_k
    )

    # 5. Apply PEFT: model = get_peft_model(model, config)
    model = get_peft_model(model, config)
    # ==========================================

    model.print_trainable_parameters()

    # Get the layer indexes of model parameters
    layer_indices = get_layer_indices(model)

    print(f"layer indices: {layer_indices.keys()}")

    # Data Setup
    # Load data (assuming pre-processed HF dataset)
    if data_path.endswith('.json') or data_path.endswith('.jsonl'):
        data = load_dataset('json', data_files=data_path)
    else:
        data = load_from_disk(data_path)

    if 'train' in data:
        train_data = data['train']
    else:
        train_data = data
    
    # Use Trainer to get a clean dataloader
    trainer = Trainer(
        model=model,
        train_dataset=train_data,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=1, # Force batch size 1 for per-sample gradients
            output_dir=output_dir,
            remove_unused_columns=False,
        ),
        data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors='pt', padding=True
        ),
    )

    train_dataloader = trainer.get_train_dataloader()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print('Computing training gradients...')
    stored_products = []

    for i, batch in enumerate(tqdm(train_dataloader)):
        # Compute gradient
        grad_vec = compute_gradients(model, batch, device)

        # TODO: Apply Hessian Inverse here
        # For now, we assume H^-1 approx identity, so product = grad_ve
        product = grad_vec.cpu()

        stored_products.append(product)

        # Break early for testing if you want
        if i >= 10: break

    print(f"stored {len(stored_products)} training vectors.")

    # Validation loop
    val_dataloader = trainer.get_eval_dataloader()

    print(f"Processing validation samples...")

    # We will store results here
    # Structure: List of dicts, where each dict is {layer_id: influence_score}
    all_val_influences = []

    for j, batch in enumerate(tqdm(val_dataloader)):
        # Compute validation gradient
        val_grad_vec = compute_gradients(model, batch, device)
        val_grad_vec = val_grad_vec.cpu()

        total_element_wise_product = torch.zero_like(val_grad_vec)

        for train_vec in stored_products:
            # Element-wise product of (H^-1 g_train) and g_val
            # We accumulate this vector
            total_element_wise_product += (train_vec * val_grad_vec)

        # Now we have a vector where each element is the total influence on that parameter
        # We aggregate these by layer

        sample_layer_influence = {}
        for layer_id, indices in layer_indices.items():
            if len(indices) > 0:
                # Sum the influence value for all parameters in this layer
                layer_vals = total_element_wise_product[indices]
                infl_val = layer_vals.sum().item()
                sample_layer_influence[layer_id] = infl_val

        all_val_influences.append(sample_layer_influence)

        # Print or save
        print(f"Val Sample {j} Influence per layer: {sample_layer_influence}")

        if j >= 2; break # limit for testing



if __name__ == "__main__":
    fire.Fire(main)

