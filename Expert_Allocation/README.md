# Expert Allocation

This pipeline performs non-uniform Mixture-of-LoRA (MoLA) fine-tuning, where each transformer layer receives a different number of LoRA experts based on its MDL-derived importance score.

See the root [README](../README.md) for the full end-to-end quickstart.

## Dependencies on `mdl/`

- `mdl/logUtility.py` consumes LayerIF values from `LayerIF_Computation/outputs/` and produces per-layer expert counts.
- Expert counts are passed to `run_all.sh` via the `number_experts` and `top_k` arguments.

## Setup

```bash
conda env create -f alphalora-train.yml   # training environment
conda env create -f alphalora-eval.yml    # evaluation environment
```

## Pipeline steps

**1. Compute Layer Influence values**

```bash
cd LayerIF_Computation
python compute_IF.py
```

Runs kronfluence on the base model to produce per-layer influence scores in `LayerIF_Computation/outputs/layerIF_values/<model>/`. Then open `LayerIF_Computation/expert_allocator.ipynb` to inspect and plot the raw IF values.

**2. Derive expert counts from MDL**

```bash
python ../mdl/logUtility.py
```

Outputs `number_experts` and `top_k` per layer. Pre-computed allocations for Mistral-7B (160 total experts) for reference:

| Method | Layer assignments (layers 0–31) |
|--------|---------------------------------|
| IF_CoLA | 1,9,10,8,9,12,10,7,9,9,9,8,6,9,6,4,7,3,2,1,3,4,2,1,4,1,1,1,1,1,1,1 |
| IF_MRPC | 1,4,6,3,4,7,8,11,5,12,10,12,7,9,7,5,8,5,4,1,6,6,1,1,6,3,1,3,1,1,1,1 |
| IF_OpenBook | 1,5,5,8,7,5,8,7,8,5,7,7,7,6,6,6,6,6,7,5,7,5,3,2,2,4,3,4,2,2,3,1 |
| MoLA(2,4,6,8) | 2,2,2,2,2,2,2,2,4,4,4,4,4,4,4,4,6,6,6,6,6,6,6,6,8,8,8,8,8,8,8,8 |
| AlphaLora | 1,3,5,4,5,5,4,4,3,4,3,2,2,3,3,4,9,4,7,7,7,7,7,7,9,7,6,8,6,7,4,3 |

**3. Train MoLA on six datasets**

Edit `run_all.sh` to set these hyperparameters, then run:

```bash
bash run_all.sh
```

| Parameter | Description |
|-----------|-------------|
| `base_model` | Path or HF hub ID of the base LLM |
| `root_data_path` | Path to the six fine-tuning datasets |
| `number_experts` | Comma-separated expert count per layer (32 values) |
| `top_k` | Comma-separated top-k value per layer (32 values) |
| `output_dir` | Where to save LoRA expert weights |

**4. Evaluate**

```bash
bash eval_all.sh
```

Ensure `mola_weights` matches the `output_dir` from training and that `number_experts`/`top_k` are identical.
