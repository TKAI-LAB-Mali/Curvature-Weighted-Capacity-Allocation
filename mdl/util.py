import json
import os
import pickle as pkl
import numpy as np

def get_IF():
    path = '/data/mdl-layerIF/Expert_Allocation/layerIF_Computation/outputs/layerIF_values/mistral-7B'
    layer_IFs = []
    for file in sorted(os.listdir(path)):
        if file.endswith('pkl'):
            results = pkl.load(open(os.path.join(path, file), 'rb'))
            layerIF_value = results['influence']['proposed'].to_numpy()
            layer_IFs.append(-1.0*np.sum(layerIF_value))
    return layer_IFs

    