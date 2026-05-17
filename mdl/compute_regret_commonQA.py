import matplotlib.pyplot as plt
import numpy as np
import util_mali as mutil
import pandas as pd
from prune import normalize_layer_qualities

# 1. Define your datasets
datasets = ['commonq', 'mrpc', 'cola', 'openbook', 'text_science_q_rebuttal']
targets = ['CommonQA (Source)', 'MRPC', 'CoLA', 'OpenBookQA', 'ScienceQA']
influence_path = "/data/mdl-layerIF/Expert_Allocation/LayerIF_Computation/outputs/layerIF_values/gemma-7b"
layer_IFs = []
for dataset in datasets:
    print(f"Dataset: {dataset}")
    dataset_layer_IF, _ = mutil.get_IF(influence_path, dataset, 'all')
    dataset_layer_IF = normalize_layer_qualities(dataset_layer_IF)
    layer_IFs.append(dataset_layer_IF)

layer_IFs = np.array(layer_IFs)
l2_distances = np.square(layer_IFs - layer_IFs[0])  # Squared L2 distance from CommonQA (index 0)
l2_sum = np.sum(l2_distances, axis=1)  # Sum across layers to get a single score drift value per dataset
print(f"Curvature Score Drifts (Squared L2 Distances) from CommonQA: {l2_sum}")




# 2. Input your calculated X-values (Squared L2 Distances)
# The distance from CommonQA to itself is exactly 0.0
# Replace the numbers below with your actual calculated L2 distances.
score_drift = l2_sum

commonQA_target_acc = np.array([81.57, 85.33, 85.52, 87.6, 93.48])
target_target_acc = np.array([81.57, 84.99, 87.34, 88.80, 94.92])
# 3. Input your calculated Y-values (Accuracy Drops in %)
# The regret from CommonQA to itself is exactly 0.0
# Replace the numbers below with your actual calculated regret values.
regret =  target_target_acc -commonQA_target_acc
print(f"Empirical Regret (Accuracy Drop %) from CommonQA: {regret} Gemama-7B") 

# --- Plot Generation ---
plt.figure(figsize=(8, 6))

# Plot the empirical data points
plt.scatter(score_drift, regret, color='#1f77b4', s=100, zorder=5, label='Empirical Regret')

corr_table = []
# Annotate each point with the dataset name
for i, txt in enumerate(targets):
    plt.annotate(txt, (score_drift[i], regret[i]), 
                 xytext=(8, 5), textcoords='offset points', fontsize=10)
    corr_table.append([txt, score_drift[i], regret[i]])

print(pd.DataFrame(corr_table, columns=['Dataset', 'Score Drift', 'Regret']))

print(corr_table)
# Create the Theoretical O(delta^2) Upper Bound
# We find a constant 'c' that creates an envelope just above your maximum regret point
c = np.max(regret[1:] / score_drift[1:]) * 1.1 
x_line = np.linspace(0, max(score_drift) * 1.1, 100)
y_line = c * x_line

plt.plot(x_line, y_line, color='red', linestyle='--', linewidth=2, 
         label=r'Theoretical $O(\delta^2)$ Bound', zorder=4)

# Formatting for publication
plt.xlabel(r'Curvature Score Drift ($||q^{(Source)} - q^{(Target)}||^2_2$)', fontsize=12)
plt.ylabel('Empirical Transfer Regret (Accuracy Drop %)', fontsize=12)
plt.title('Transfer Regret vs. Curvature Score Drift (Source: CommonsenseQA), Gemma-7B', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(loc='upper left', fontsize=11)
plt.tight_layout()

# Save as a PDF for easy inclusion in your LaTeX rebuttal document
# plt.savefig('transfer_regret_theorem4.pdf', format='pdf', dpi=300)
plt.savefig('transfer_regret_theorem4_table.pdf', format='pdf', dpi=300)
plt.show()