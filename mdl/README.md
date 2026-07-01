# mdl/ — MDL Core

This directory implements the Minimum Description Length (MDL) scoring framework shared by both pipelines. It computes curvature-weighted layer importance scores and translates them into either per-layer expert counts (expert allocation) or per-layer sparsity ratios (pruning).

## Scripts

| Script | Purpose |
|--------|---------|
| `logUtility.py` | MDL → per-layer expert counts for MoLA fine-tuning. Reads LayerIF values from `Expert_Allocation/LayerIF_Computation/outputs/` and outputs expert/top-k assignments. Adjust `rho` (line 188) to control the total expert budget. |
| `pruning.py` | MDL → per-layer sparsity ratios. Reads LayerIF scores and model metadata, writes ratio JSONs to `mdl/data/`. Used by `LayerIF_Pruning_New/main.py`. |
| `my_layer_influence.py` | Computes Layer Influence Functions via [kronfluence](https://github.com/pomonam/kronfluence). Entry point for computing the raw IF scores that feed into both pipelines. |
| `util.py` | Core MDL utilities: description length computation, budget allocation math. |
| `util_mali.py` | Extended utilities for the MaLI variant of the MDL scoring. |
| `get_B.py` | Curvature (Fisher/Hessian diagonal) estimation helpers. |
| `model_profiling.py` | Model parameter counting and layer-dimension utilities. |

## Data flow

```
Expert_Allocation/LayerIF_Computation/compute_IF.py
        │
        ▼  (per-layer IF scores)
Expert_Allocation/LayerIF_Computation/outputs/layerIF_values/<model>/
        │
        ├──► mdl/logUtility.py  ──► expert counts per layer
        │                              (used by Expert_Allocation/run_all.sh)
        │
        └──► mdl/pruning.py     ──► mdl/data/*.json  (pruning ratio JSONs)
                                       (used by LayerIF_Pruning_New/main.py)
```

## Pre-computed results

`mdl/data/` — per-layer MDL pruning ratios for Mistral-7B and Gemma-7B across multiple sparsity targets. These are the paper's core output and are consumed directly by `LayerIF_Pruning_New/run_mistral_mdl.sh`.

`mdl/data_dp/` — depth-prior variant pruning ratios, used by `LayerIF_Pruning_New/run_mistral_mdl_depth_prior.sh`.

`mdl/visualizations/` — JSON data backing the paper's expert allocation and pruning figures.
