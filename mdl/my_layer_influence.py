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

from kronfluence.analyzer import Analyzer, prepare_model
from kronfluence.task import Task
from kronfluence.utils.dataset import DataLoaderKwargs
from kronfluence.arguments import FactorArguments, ScoreArguments

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
import logging # Add logging import

os.environ['WANDB_DISABLED'] = 'false'

import random
seed = 42
random.seed(seed)
torch.manual_seed(seed)


class MistralTask(Task):
    def compute_train_loss(self, batch, model, sample = False):
        """
        computes the loss for the training data
        """
        # return super().compute_train_loss(batch, model, sample)
        # Move batch to device (kronfluence passes the model's device implicitely usually, but good to be safe if batch isn't already there)
        device = next(model.parameters()).device
        batch = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}

        # Forward pass
        # Your model signature might be standard, but this ensures it works with your dict
        outputs = model(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            labels=batch['labels'],
            output_router_logits=False
            )
        
        # Return loss
        # If sample=True, kronfluence might expect per-sample loss, 
        # but usually for the main flow it expects a scalar mean/sum. 
        # For influence, we typically minimize the loss
        return outputs.loss
    
    def compute_measurement(self, batch, model):
        """
        Computes the metric for the validation/query points.
        For influence functions, this is usually the same as the training loss
        (we want to see how training points affect validation loss)
        """
        return super().compute_measurement(batch, model)


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
    data_path: str = "/data/mdl-layerIF/Expert_Allocation/datasets/glue_cola_all.hf",
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

    device_map = {"": 0}
    
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

    # Tokenization helpers
    cutoff_len = 256
    train_on_inputs = True
    add_eos_token = True
    prompter = Prompter("alpaca") 

    def tokenize(prompt, add_eos_token=True):
        result = tokenizer(
            prompt,
            truncation=True,
            max_length=cutoff_len,
            padding=False,
            return_tensors=None,
        )
        if (
            result["input_ids"][-1] != tokenizer.eos_token_id
            and len(result["input_ids"]) < cutoff_len
            and add_eos_token
        ):
            result["input_ids"].append(tokenizer.eos_token_id)
            result["attention_mask"].append(1)

        result["labels"] = result["input_ids"].copy()
        return result

    def generate_and_tokenize_prompt(data_point):
        full_prompt = prompter.generate_prompt(
            data_point["instruction"],
            data_point["input"],
            data_point["output"],
        )
        tokenized_full_prompt = tokenize(full_prompt)
        if not train_on_inputs:
            user_prompt = prompter.generate_prompt(
                data_point["instruction"], data_point["input"]
            )
            tokenized_user_prompt = tokenize(
                user_prompt, add_eos_token=add_eos_token
            )
            user_prompt_len = len(tokenized_user_prompt["input_ids"])

            if add_eos_token:
                user_prompt_len -= 1

            tokenized_full_prompt["labels"] = [
                -100
            ] * user_prompt_len + tokenized_full_prompt["labels"][
                user_prompt_len:
            ]
        return tokenized_full_prompt

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
        # number_experts=number_experts,
        # top_k=top_k
    )
    config.number_experts = number_experts
    config.top_k = top_k

    # 5. Apply PEFT: model = get_peft_model(model, config)
    model = get_peft_model(model, config)
    
    # Ensure model is in float16
    model = model.to(torch.float16)
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
    
    train_data = train_data.map(generate_and_tokenize_prompt, remove_columns=train_data.column_names)

    if 'validation' in data:
        val_data = data['validation']
    elif 'test' in data:
        val_data  = data['test']
    else:
        # If no validation set, split train data
        print("No validation set found. Splitting train data...")
        split = train_data.train_test_split(test_size=0.1)
        train_data = split['train']
        val_data = split['test']
    
    if 'validation' not in data and 'test' not in data:
         # If we split, we already tokenized train_data which included val_data
         pass
    else:
         val_data = val_data.map(generate_and_tokenize_prompt, remove_columns=val_data.column_names)
    
    # Use Trainer to get a clean dataloader
    trainer = Trainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=1, # Force batch size 1 for per-sample gradients
            per_device_eval_batch_size=1,
            output_dir=output_dir,
            remove_unused_columns=False,
        ),
        data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors='pt', padding=True
        ),
    )

    train_dataloader = trainer.get_train_dataloader()
    val_dataloader = trainer.get_eval_dataloader()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Define the Task for kronfluence library
    task = MistralTask()

    # Prepare the model
    # Kronfluence needs to wrap the model to track gradients. 
    # You must specify the target modules (your LoRA layers). 
    # Based on your config: "q_proj", "v_proj", etc. However, in the PEFT model, these are usually name like "base_model.model.layers.0.self_attn.q_proj.lora_A.default"
    # You can find the module names by printing model.named_modules()

    # Helper to find LoRA module names
    # TODO: Check if this snippet is needed? Because we need IF values per layer regardless of the lora expert weights
    target_module_names = []
    for name, module in model.named_modules():
        if 'lora_' in name and 'default' in name: # Adjust based on your exact structure
            target_module_names.append(name)

    print(f"Analyzing {len(target_module_names)} LoRA modules.")

    model = prepare_model(model, task)
    
    # Initialize Analyzer
    analyzer = Analyzer(
        analysis_name='mistral_lora_influence',
        model=model,
        task=task,
        log_level=logging.INFO,
        disable_model_save=True,
        cpu=False
    )

    dataloader_kwargs = DataLoaderKwargs(collate_fn=trainer.data_collator)

    # Compute Factors (EK-FAC)
    # fit_all_factors computes both Covariance and Lambda matrices needed for influence
    factor_args = FactorArguments(
        covariance_module_partitions=64,
        lambda_module_partitions=64
    )
    analyzer.fit_all_factors(
        factors_name='my_factors',      # Choose a consistent name
        dataset=train_data,             # FIX: argument name is 'dataset'
        per_device_batch_size=1,
        dataloader_kwargs=dataloader_kwargs,
        overwrite_output_dir=True,
        factor_args=factor_args
    )

    # compute Influence Scores
    score_args = ScoreArguments(module_partitions=64)
    scores = analyzer.compute_pairwise_scores(
        scores_name='pairwise_scores',
        factors_name='my_factors',      # FIX: Must match the name used in fit_all_factors
        query_dataset=val_data,         # FIX: argument is 'query_dataset'
        train_dataset=train_data,       # FIX: argument is 'train_dataset'
        per_device_query_batch_size=1,
        per_device_train_batch_size=1,
        dataloader_kwargs=dataloader_kwargs,
        overwrite_output_dir=True,
        score_args=score_args
    )

    save_path = os.path.join(output_dir, "raw_influence_scores.pt")
    torch.save(scores, save_path)
    print(f'Saved raw scores to {save_path}')


    # Aggregate scores per layer
    layer_influence_scores = {} # {layer_id: tensor[num_query, num_train]}

    for module_name, module_scores in scores.items():
        # Extract layer ID from module name
        # Example: "base_model.model.layers.0.self_attn..."
        parts = module_name.split('.')
        layer_id = -1
        for part in parts:
            if part.isdigit():
                layer_id = int(part)
                break
        if layer_id != -1:
            if layer_id not in layer_influence_scores:
                layer_influence_scores[layer_id] = torch.zeros_like(module_scores)

            # Add this module's influence to the layer's total influence
            layer_influence_scores[layer_id] += module_scores

    # Save the aggregated scores
    aggregated_save_path = os.path.join(output_dir, 'layer_wise_influence.pt')
    torch.save(layer_influence_scores, aggregated_save_path)
    print(f'saved layer-wise scores to {aggregated_save_path}')

    # Optional: Save as JSON (if small enough)
    # convert tensors to lists
    json_scores = {k: v.tolist() for k, v in layer_influence_scores.items()}
    with open(os.path.join(output_dir, "layer_wise_influence.json"), 'w') as f:
        json.dump(json_scores, f)


    """
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
    """


if __name__ == "__main__":
    fire.Fire(main)

