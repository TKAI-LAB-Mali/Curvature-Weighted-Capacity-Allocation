import numpy as np
import matplotlib.pyplot as plt

# --- Data Preparation ---
datasets = ['cola', 'mrpc', 'commonq', 'openbook', 'scienceQ']
kappas = [1, 2, 3, 4]

# Averages
mag_avg = [33.34, 33.20, 33.23, 33.136]
wan_avg = [49.47, 49.49, 49.364, 48.63]
spa_avg = [49.40, 49.008, 49.30, 48.83]

layerIF_avg = [33.91, 52.3, 50.79]

# Full Breakdown Data
kappa_data = {
    1: {'Magnitude': [33.03, 33.49, 33.51, 33.52, 33.15],
        'Wanda': [48.36, 52.06, 50.26, 48.09, 48.58],
        'Sparsegpt': [49.28, 50.77, 50.18, 48.11, 48.66]},
    2: {'Magnitude': [33.21, 32.96, 33.44, 33.03, 33.36],
        'Wanda': [47.37, 52.17, 50.51, 48.93, 48.48],
        'Sparsegpt': [46.97, 50.99, 49.75, 48.15, 49.18]},
    3: {'Magnitude': [32.97, 33.34, 33.61, 33.00, 33.25],
        'Wanda': [45.08, 54.20, 50.11, 48.48, 48.95],
        'Sparsegpt': [46.38, 52.12, 50.70, 47.89, 49.43]},
    4: {'Magnitude': [32.72, 33.27, 33.63, 32.69, 33.37],
        'Wanda': [45.20, 52.52, 49.11, 47.84, 48.46],
        'Sparsegpt': [45.22, 51.51, 50.49, 48.44, 48.29]}
}

# --- 1. Plotting Averages ---
fig, ax = plt.subplots(figsize=(8, 5.5))
ax.plot(kappas, mag_avg, marker='o', label='Magnitude', color='tab:blue', linewidth=2)
ax.plot(kappas, wan_avg, marker='s', label='Wanda', color='tab:orange', linewidth=2)
ax.plot(kappas, spa_avg, marker='^', label='Sparsegpt', color='tab:green', linewidth=2)


ax.axhline(y=layerIF_avg[0], color='tab:blue', linestyle='--', linewidth=1.5, alpha=0.8, 
           label=f'LayerIF Computed Mag ({layerIF_avg[0]})')
ax.axhline(y=layerIF_avg[1], color='tab:orange', linestyle='--', linewidth=1.5, alpha=0.8, 
           label=f'LayerIF Computed Wanda ({layerIF_avg[1]})')
ax.axhline(y=layerIF_avg[2], color='tab:green', linestyle='--', linewidth=1.5, alpha=0.8, 
           label=f'LayerIF Computed Sparsegpt ({layerIF_avg[2]})')

plt.title('Average Performance vs Kappa')
plt.xlabel('Kappa Value', fontsize=15)
plt.ylabel('Average Zero-Shot Accuracy (%)', fontsize=15)
plt.xticks(kappas, fontsize=15)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('kappa_average.pdf', dpi=300)
plt.close()

# --- 2. Plotting Full Breakdown (2x2 Grid) ---
fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharey=True)
fig.suptitle('Performance Breakdown by Dataset and Method (Kappa 1-4)', fontsize=16)

x = np.arange(len(datasets))
width = 0.25

for i, ax in enumerate(axes.flatten()):
    k_val = i + 1
    b1 = ax.bar(x - width, kappa_data[k_val]['Magnitude'], width,
            label='Magnitude', color='tab:blue', edgecolor='black', hatch='//')
    b2 = ax.bar(x, kappa_data[k_val]['Wanda'], width, 
           label='Wanda', color='tab:orange', edgecolor='black', hatch='++')
    b3 = ax.bar(x + width, kappa_data[k_val]['Sparsegpt'], width, 
           label='Sparsegpt', color='tab:green', edgecolor='black', hatch='..')
    
    ax.bar_label(b1, fmt='%.2f', padding=3, fontsize=11, rotation=45)
    ax.bar_label(b2, fmt='%.2f', padding=3, fontsize=11, rotation=45)
    ax.bar_label(b3, fmt='%.2f', padding=3, fontsize=11, rotation=45)
    
    ax.set_title(f'Kappa = {k_val}')
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, fontsize=15)
    ax.set_ylabel('Zero-Shot Accuracy (%)', fontsize=15)

    ax.set_ylim(0, 75)
    
    # Only put the legend in the first subplot to avoid clutter
    if i == 0:
        ax.legend(loc='lower right')
        
    ax.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig('kappa_breakdown.pdf', dpi=300)
plt.close()

print("Kappa plots generated successfully.")