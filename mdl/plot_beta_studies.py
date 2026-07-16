import matplotlib.pyplot as plt
import numpy as np

# --- Data Preparation ---
datasets = ['cola', 'mrpc', 'commonq', 'openbook', 'scienceQ']

# LayerIF is constant across all Betas
layerif_data = [87.15, 84.41, 82.64, 88.40, 94.69]
layerif_avg = 87.46

# MDL (+ve) fluctuates
mdl_b1 = [87.34, 82.37, 82.80, 86.60, 94.56]
mdl_b2 = [86.77, 83.30, 79.27, 87.60, 94.38]
mdl_b3 = [87.34, 84.99, 81.57, 88.80, 94.92]
mdl_avgs = [86.734, 86.265, 87.525]
betas = [1, 2, 3]

# --- 1. Plotting Consolidated Averages ---
plt.figure(figsize=(7, 4))
plt.plot(betas, mdl_avgs, marker='s', label='MDL (+ve)', color='tab:blue', linewidth=2)
# Plot LayerIF as a constant horizontal threshold line
plt.axhline(y=layerif_avg, color='tab:gray', linestyle='--', linewidth=2, label='LayerIF (Constant)')

plt.title('Average Performance vs Beta')
plt.xlabel('Beta')
plt.ylabel('Average Accuracy (%)')
plt.xticks(betas)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('beta_average_consolidated.pdf', dpi=300)
plt.close()

# --- 2. Plotting Consolidated Full Breakdown ---
fig, ax = plt.subplots(figsize=(12, 6))
fig.suptitle('Performance Breakdown by Dataset: LayerIF vs MDL (+ve)', fontsize=20)

x = np.arange(len(datasets))
width = 0.2  # Thinner bars to fit 4 per group

# Grouping the bars together per dataset
bar_base = ax.bar(x - 1.5*width, layerif_data, width, 
       label='LayerIF (Baseline)', color='tab:gray', edgecolor='black', hatch='//')
bar_mdl1 = ax.bar(x - 0.5*width, mdl_b1, width,
       label='MDL (+ve) Beta=1', color='tab:blue', edgecolor='black', hatch='xx')

bar_mdl2 = ax.bar(x + 0.5*width, mdl_b2, width, label='MDL (+ve) Beta=2',
        color='tab:orange', edgecolor='black', hatch='..')
bar_mdl3 = ax.bar(x + 1.5*width, mdl_b3, width, label='MDL (+ve) Beta=3',
        color='tab:green', edgecolor='black', hatch='\\\\')

ax.bar_label(bar_base, fmt='%.2f', padding=3, fontsize=12, rotation=45)
ax.bar_label(bar_mdl1, fmt='%.2f', padding=3, fontsize=12, rotation=45)
ax.bar_label(bar_mdl2, fmt='%.2f', padding=3, fontsize=12, rotation=45)
ax.bar_label(bar_mdl3, fmt='%.2f', padding=3, fontsize=12, rotation=45)

ax.set_xticks(x)
ax.set_xticklabels(datasets, fontsize=15)
ax.set_ylabel('Accuracy (%)', fontsize=15)
# Place legend outside to avoid obscuring the tall scienceQ bars
ax.set_ylim(75, 100)  # Adjust y-axis to better visualize differences
ax.legend(loc='upper left', fontsize=12)
ax.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig('beta_breakdown_consolidated.pdf', dpi=300)
plt.close()

print("Consolidated plots generated successfully.")