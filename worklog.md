# Work log

This file contains information about the progress of the work and notes for later reference.

## Expert Allocation Experiments

### Steps of implementation

1. Go to [mdl/logUtility.py](mdl/logUtility.py) for expert allocation experiments
2. Adjust scaling factor for budget i.e. `rho` at [mdl/logUtility.py line 188](mdl/logUtility.py#L188) to set the desired budget of number of experts for your dataset of choice. In our experiments `rho` is chosen so that the total experts turn out to 160. This is to follow the convention of layerIF paper.
3. Check if the folder paths are valid
4. set your choice of IF values at [mdl/logUtility.py line 147](mdl/logUtility.py line 147)
5. Execute the program to get `number of experts` and `top_k` values per layer for each dataset

### Hardware requirements for Expert allocation experiment
