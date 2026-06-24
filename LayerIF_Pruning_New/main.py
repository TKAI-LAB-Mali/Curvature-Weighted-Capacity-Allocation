import argparse
import ast
import os
import sys
import json
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import time
from functools import partialmethod
from tqdm import tqdm
tqdm.__init__ = partialmethod(tqdm.__init__, mininterval=60.0)


from lib.prune import prune_wanda, prune_sparsegpt, prune_magnitude, prune_wanda_ww, prune_sparsegpt_ww, prune_magnitude_ww, check_sparsity
from lib.eval import eval_ppl, eval_zero_shot
from lib.esd_utils import get_esd_metrics
from analysis.utils import get_weight_norms, get_activation_norms, save_norms


def get_llm(model, cache_dir="llm_weights"):
    model = AutoModelForCausalLM.from_pretrained(
        model,
        torch_dtype = torch.float16,
        cache_dir = cache_dir,
        low_cpu_mem_usage=True,
        device_map = "auto"
    )
    
    model.seqlen = 2048
    return model


def main():
   
    parser = argparse.ArgumentParser()  
    parser.add_argument('--model', type=str, help="model type")
    parser.add_argument('--seed', type=int, default=0, help='Seed for sampling the calibration data')
    parser.add_argument('--nsamples', type=int, default=128, help='Number of calibration samples')
    parser.add_argument('--sparsity_ratio', type=float, default=0, help='Sparsity level')
    parser.add_argument('--prune_method', type=str)
    parser.add_argument('--sparsity_type', type=str, default="unstructured", help='Structured pruning for N:M')
    parser.add_argument('--cache_dir', default="llm_weights", type=str)
    parser.add_argument('--save', type=str, default=None, help='Path to save results')
    parser.add_argument('--save_model', type=str, default=None, help='Path to save the pruned model.')
    parser.add_argument('--use_variant', action="store_true", help="whether to use the wanda variant described in the wanda paper appendix")
    
    # params for WW
    parser.add_argument("--ww_metric", default="alpha_peak", type=str, help="the WW-based metric to ues.")
    parser.add_argument("--ww_metric_cache", default="./data/llama-7b-hf")
    parser.add_argument("--epsilon", default=0.3, type=float, help="for pruning ratio allocation.")
    parser.add_argument("--mapping_type", default="block_wise", type=str, help="mapping type for pruning ratios allocation.")
    # evaluation benchmark
    parser.add_argument("--eval_zero_shot", action="store_true", help="evaluation on zero-shot tasks.")
    parser.add_argument("--dataset", type=str, choices=["boolq", "rte","hellaswag","winogrande", "arc_easy","arc_challenge", "openbookqa"] )
    parser.add_argument("--eval_wikitext", type=bool, default=True, help="evaluation on wikitext.")
    
    
    # mdl
    parser.add_argument('--mdl', default=None, help='specify dataset name to use ratios obtained from MDL') 
    parser.add_argument('--mdl_path', default=None, help='specify path to the MDL sparsity ratios file')
    parser.add_argument('--ratio_name', type=str, default="Unknown_ratio", help='a short name for the pruning ratio set, used for plotting legends.')
    parser.add_argument('--batch_size', type=int, default=4, help='batch size for pruning and evaluation.')

    # Norm tracking
    parser.add_argument("--track_norms", action="store_true",
                        help="Compute and save weight + activation norms before and after pruning.")
    parser.add_argument("--norm_n_samples", type=int, default=4,
                        help="Number of prompts to use for activation norm estimation.")
    parser.add_argument("--norm_seq_len", type=int, default=128,
                        help="Sequence length for activation norm forward pass.")

    
    
    args = parser.parse_args()
    
    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)
    start_time = time.time()

    # if args.rho_l:
    #     rho_l = [x for x in args.rho_l for _ in range(7)]

    if args.mdl_path and '--eval_wikitext' not in sys.argv:
        args.eval_wikitext = False
    
    # get the layerwise metric values of the model
    if "ww" in args.prune_method and not os.path.exists("{}/{}.npy".format(args.ww_metric_cache, args.ww_metric)):
        metric_values = get_esd_metrics(args.model, args.ww_metric, args.cache_dir)
        np.save(f"{args.ww_metric_cache}/{args.ww_metric}.npy", metric_values)



    model_name = args.model.split("/")[-1]
    model = get_llm(args.model, args.cache_dir)
    model.eval()

    # Handling n:m sparsity
    prune_n, prune_m = 0, 0
    if args.sparsity_type != "unstructured":
        prune_n, prune_m = map(int, args.sparsity_type.split(":"))
    
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)
    
    device = torch.device("cuda:0")
    if any(s in args.model for s in ["7b", "13b", "30b", "65b", "70b"]): # for 30b or 65b or 70b, we use device_map to load onto multiple GPUs, thus the processing here.
        device = model.hf_device_map["lm_head"]
    print("use device ", device)

    # ------------------------------------------------------------------
    # Norm tracking — BEFORE pruning
    # ------------------------------------------------------------------
    if args.mdl_path:
        if args.track_norms:
            print("[norms] Computing weight and activation norms BEFORE pruning...")
            weight_norms_before = get_weight_norms(model)
            activation_norms_before = get_activation_norms(
                model, tokenizer, device,
                n_samples=args.norm_n_samples,
                seq_len=args.norm_seq_len,
            )
            # Save immediately so a crash after pruning doesn't lose the baseline
            save_norms(weight_norms_before, activation_norms_before, args.save, 
                    tag=f"before_{args.prune_method}_{args.ratio_name}")
            print(f"[norms] Saved before-pruning norms to {args.save}")

    
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    # ------------------------------------------------------------------
    # Pruning
    # ------------------------------------------------------------------
    

    if args.sparsity_ratio != 0:
        print("pruning starts")

        mdl_sparsity_ratios = None
        if args.mdl:
            print(f"reading mdl sparsity ratios")
            with open(f'/data/mdl-layerIF/mdl/data/prune_ub-0.51-{args.mdl}.json', 'r') as file:
                mdl_sparsity_ratios = json.load(file)
            mdl_sparsity_ratios = [num for num in mdl_sparsity_ratios for _ in range(7)]    # each layer in LLM has 7 weights
        #======================================
        elif args.mdl_path:
            print(f"reading mdl sparsity ratios from {args.mdl_path}")
            with open(args.mdl_path, 'r') as file:
                mdl_data = json.load(file)
                mdl_sparsity_ratios = mdl_data["rho_l"]
            mdl_sparsity_ratios = [num for num in mdl_sparsity_ratios for _ in range(7)]    # each layer in LLM has 7 weights
        #======================================

        if "OWL" in args.prune_method:
            owl_path = os.path.join(args.ww_metric_cache, f"{args.ww_metric}.txt")
            print(f"OWL path: {owl_path}")
            if os.path.exists(owl_path):
                print(f"Reading OWL scores from {owl_path}")
                with open(owl_path, "r") as file:
                    mdl_sparsity_ratios = ast.literal_eval(file.read())
                print(f"OWL scores: {mdl_sparsity_ratios}")

        
        # Uniform pruning
        if args.prune_method == "wanda" or args.prune_method == "wanda_OWL":
            prune_wanda(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m, ratios=mdl_sparsity_ratios)

        elif args.prune_method == "magnitude" or args.prune_method == "magnitude_OWL":
            print(f"ratios: {mdl_sparsity_ratios}")
            prune_magnitude(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m, ratios=mdl_sparsity_ratios)

        elif args.prune_method == "sparsegpt" or args.prune_method == "sparsegpt_OWL":
            prune_sparsegpt(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m, ratios=mdl_sparsity_ratios)

        ################################################
        # Pruning with our layerwise pruning ratios
        elif args.prune_method == "wanda_ww":
            prune_wanda_ww(args, model, tokenizer, device, ratios=mdl_sparsity_ratios)

        elif args.prune_method == "magnitude_ww":
            prune_magnitude_ww(args, model, tokenizer, device, ratios=mdl_sparsity_ratios)

        elif args.prune_method == "sparsegpt_ww":
            prune_sparsegpt_ww(args, model, tokenizer, device, ratios=mdl_sparsity_ratios)
    
    # ------------------------------------------------------------------
    # Norm tracking — AFTER pruning
    # ------------------------------------------------------------------
    if args.mdl_path:
        if args.track_norms:
            print("[norms] Computing weight and activation norms AFTER pruning...")
            weight_norms_after = get_weight_norms(model)
            activation_norms_after = get_activation_norms(
                model, tokenizer, device,
                n_samples=args.norm_n_samples,
                seq_len=args.norm_seq_len,
            )
            save_norms(weight_norms_after, activation_norms_after, args.save, 
                    tag=f"after_{args.prune_method}_{args.ratio_name}")
            print(f"[norms] Saved after-pruning norms to {args.save}")
    
    
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    sparsity_ratio = check_sparsity(model)
    
    if not os.path.exists(args.save):
        os.makedirs(args.save)
    
    if args.eval_wikitext:
        ppl_test = eval_ppl(args, model, tokenizer, device)
        print(f"wikitext perplexity {ppl_test}")

        save_filepath = os.path.join(args.save, f"perplexity_{args.prune_method}_sparsity_{args.sparsity_ratio}.txt")
        with open(save_filepath, "w") as f:
            print("method\tactual_sparsity\tppl_test", file=f, flush=True)
            print(f"{args.prune_method}\t{sparsity_ratio:.4f}\t{ppl_test:.4f}", file=f, flush=True)
    
    # zero-shot tasks evaluation
    if args.eval_zero_shot:
        accelerate=True
        
        # if "7b" in args.model or "13b" in args.model or "30b" in args.model or "65b" in args.model or "70b" in args.model or 'Llama-2-13b-hf' in args.model:
        #     accelerate=True

        if args.dataset:
            print('here')
            print(args.dataset)
            task_list=[]
            task_list.append(args.dataset)
            #task_list=list(args.dataset)
        else:    
            task_list = ["boolq", "rte","hellaswag","winogrande", "arc_easy","arc_challenge", "openbookqa"]
        num_shot = 0
        results = eval_zero_shot(args.model, model, tokenizer, task_list, num_shot, accelerate)
        print("zero_shot evaluation results")

          
        if args.mdl_path:
            task_acc = []
            for task in task_list:
                task_acc.append(results['results'][task]['acc'])
            
            results["avg_acc"] = np.mean(task_acc).item()
            
            save_filepath = os.path.join(args.save, f"zero_shot_{args.prune_method}_sparsity_{args.sparsity_ratio}_epsilon_{args.epsilon}.json")
            with open(save_filepath, "w") as f:
                json.dump({
                    f"{args.prune_method}": results,
                }, f, indent=4)
        else:
            # print(f"MDL pruning ratios from dataset {args.mdl} were used.")
            # print(results)

            elapsed_time = time.time() - start_time
            save_filepath = os.path.join(args.save, f"zero_shot_{args.prune_method}_sparsity_{args.sparsity_ratio}_epsilon_{args.epsilon}_IFdata_{args.mdl}.txt")
            with open(save_filepath, "w") as f:
                print(f"{args.prune_method}:\n{results}", file=f, flush=True) 
                print(f"Total time: {elapsed_time/60.0:.2f} minutes")
    
    # save model if needed.    
    if args.save_model:
        save_model_path = os.path.join(args.save_model, f"{args.prune_method}_{args.sparsity_ratio}")
        if not os.path.exists(save_model_path):
            os.makedirs(save_model_path)
        model.save_pretrained(save_model_path)
        tokenizer.save_pretrained(save_model_path)
    
    
if __name__ == '__main__':
    main()
