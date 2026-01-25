# Work log

This file contains information about the progress of the work and notes for later reference.


## Expert Allocation Experiments
- the compute_IF.py did not produce IF values for initial layers on `text_science_q_rebuttal` and `commonq` datasets. Therefore, compute_IF.py is being executed again for those datasets.
- the output of run_all.sh for `cola` dataset are stored in `/data/mdl-layerIF/Expert_Allocation/layerIF_Computation/outputs/layerIF_values/mistral-7B`
- The outputs of `mrpc`, `openbook`. `commonq` and `text_science` are stored in `/data/mdl-layerIF/Expert_Allocation/layerIF_Computation/outputs/layerIF_values/Mistral-7B-v0.1`
- `eval.sh` now performed on cola, mrpc, openbook