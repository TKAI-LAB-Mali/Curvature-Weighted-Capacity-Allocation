# Steps for experiments

### Environment
`source bin/activate`

## Expert Allocation - (log utility)
1. Get number of experts with [Expert_Allocation/expert_number.py](Expert_Allocation/expert_number.py). Follow [Expert_Allocation/README.md](Expert_Allocation/README.md)
2. Obtaining number of experts based on layerIF - [Expert_Allocation/layerIF_Computation/compute_IF.py](Expert_Allocation/layerIF_Computation/compute_IF.py)
3. Obtain number of experts based on MDL - [mdl/logUtility.py](mdl/logUtility.py)
4. Train with number of experts - [Expert_Allocation/run_all.sh](Expert_Allocation/run_all.sh) or [Expert_Allocation/run_all_mola.sh](Expert_Allocation/run_all_mola.sh)
5. Evaluate on datasets - [Expert_Allocation/eval_all.sh](Expert_Allocation/eval_all.sh)

## Pruning
1. Testing pruning using LayerIF - [LayerIF_Pruning_New/README.md](LayerIF_Pruning_New/README.md)
2. Obtain layer wise pruning ratios using MDL - [mdl/pruning.py](mdl/pruning.py) set number of experts_per_layer and model_metadata parameters
3. Evaluation - LayerIF_Pruning_New/run_*.sh