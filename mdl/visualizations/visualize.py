import matplotlib.pyplot as plt
import numpy as np
import json

# Set global font sizes
plt.rcParams.update({
    'font.size': 16,           # Default font size
    'axes.titlesize': 18,      # Title font size
    'axes.labelsize': 16,      # X and Y label font size
    'xtick.labelsize': 16,     # X tick labels
    'ytick.labelsize': 16,     # Y tick labels
    'legend.fontsize': 16,     # Legend font size
})

def read_mistral_expert_alloc(path):
    with open(path, 'r') as f:
        json_data = json.load(f)
    
    dataset = []
    all_layerIF = []
    all_mdl = []
    negative_layerIF = []
    negative_mdl = []

    for sample in json_data:
        dataset.append(sample['Dataset'].split('-')[-1])
        all_layerIF.append(float(sample['all layerIF']))
        all_mdl.append(float(sample['all mdl']))
        negative_layerIF.append(float(sample['negative layerIF']))
        negative_mdl.append(float(sample['negative mdl']))
    return dataset, all_layerIF, all_mdl, negative_layerIF, negative_mdl

def read_pruning(path):
    with open(path, 'r') as f:
        json_data = json.load(f)

    dataset = []
    magnitude = []
    wanda = []
    sparsegpt = []
    for sample in json_data:
        dataset.append(sample['Calibration dataset'])
        magnitude.append(float(sample['Magnitude']))
        wanda.append(float(sample['Wanda']))
        sparsegpt.append(float(sample['SparseGPT']))
    return dataset, magnitude, wanda, sparsegpt

def read_gemma_expert_alloc(path):
    with open(path, 'r') as f:
        json_data = json.load(f)

    dataset = []
    layerIF = []
    mdl = []
    for sample in json_data:
        dataset.append(sample['Dataset'])
        layerIF.append(float(sample['LayerIF']))
        mdl.append(float(sample['MDL']))

    return dataset, layerIF, mdl

def plot_mistral(dataset, all_layerIF, all_mdl, negative_layerIF, negative_mdl, title):
    x = np.arange(len(dataset)) # positions on x axis
    width = 0.2     # width of each bar

    fig, ax = plt.subplots(figsize=(14, 7))

    # create 4 bars per x position
    bars1 = ax.bar(x - 1.5*width, all_layerIF, width, 
                   label='layerIF (all)', hatch='//', edgecolor='black')
    bars2 = ax.bar(x - 0.5*width, all_mdl, width, 
                   label='mdl (all, our)', hatch='//', edgecolor='black')
    bars3 = ax.bar(x + 0.5*width, negative_layerIF, width,
                   label='layerIF (+ve)', hatch='..', edgecolor='black')
    bars4 = ax.bar(x + 1.5*width, negative_mdl, width, 
                   label='mdl (+ve, our)', hatch='..', edgecolor='black')

    # Set axis limits BEFORE adding labels so we can clip
    y_min, y_max = 75, 91
    ax.set_ylim(y_min, y_max)
    ax.set_yticks(range(y_min, y_max + 1, 2))

    # Add value labels on top of each bar
    for bars in [bars1, bars2, bars3, bars4]:
        for bar in bars:
            height = bar.get_height()
            if height >= y_min:
                # Normal: label just above the bar
                ax.text(bar.get_x() + bar.get_width() / 2., height + 0.2,
                        f'{height:.1f}', ha='center', va='bottom', fontsize=10,
                        rotation=60)
            else:
                # Bar is clipped: place label at y_min so it's still visible
                ax.text(bar.get_x() + bar.get_width() / 2., y_min + 0.2,
                        f'{height:.1f}', ha='center', va='bottom', fontsize=10,
                        rotation=60, color='red')

    # Labels and formatting
    ax.set_xlabel('Datasets')
    ax.set_ylabel('Accuracies')
    ax.set_xticks(x)
    ax.set_xticklabels(dataset, fontsize=14)
    ax.tick_params(axis='y', labelsize=14)
    legend = ax.legend(fontsize=12, loc='upper left', 
                       bbox_to_anchor=(0.0, 1.15), ncol=4)
    # Make legend text bold
    for text in legend.get_texts():
        text.set_fontweight('bold')

    plt.savefig('mistral_expert_alloc.pdf', bbox_inches='tight', pad_inches=0.1)
    plt.close()

def plot_gemma(dataset, layerIF, mdl, title):
    x = np.arange(len(dataset)) # positions on x axis
    width = 0.2     # width of each bar

    fig, ax = plt.subplots(figsize=(12, 6))

    # create 4 bars per x position
    # bars1 = ax.bar(x - 1.5*width, all_layerIF, width, 
    #                label='layerIF (all)', hatch='//', edgecolor='black')
    bars2 = ax.bar(x - 0.5*width, layerIF, width, 
                   label='layerIF (+ve)', edgecolor='black')
    bars3 = ax.bar(x + 0.5*width, mdl, width,
                   label='mdl (+ve, our)', edgecolor='black')
    # bars4 = ax.bar(x + 1.5*width, negative_mdl, width, 
    #                label='mdl (+ve)', hatch='..', edgecolor='black')

    # Add value labels on top of each bar
    for bars in [bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height + 0.2,
                    f'{height:.1f}', ha='center', va='bottom', fontsize=10,
                    rotation=60)

    # Labels and formatiing
    ax.set_xlabel('Datasets', fontsize=16)
    ax.set_ylabel('Accuracies', fontsize=16)
    # ax.set_title(title, fontsize=18)
    ax.set_xticks(x)
    ax.set_xticklabels(dataset, fontsize=14)
    ax.set_ylim(75, 100) # set y-axis range from 0 to 100
    ax.set_yticks(range(75, 101, 2))  # Ticks at 60, 65, 70, 75, 80, 85, 90
    ax.tick_params(axis='y', labelsize=14)
    legend = ax.legend(fontsize=14, loc='upper left')
    # Make legend text bold
    for text in legend.get_texts():
        text.set_fontweight('bold')
    plt.tight_layout(rect=[0, 0, 1, 1])
    plt.savefig('gemma_expert_alloc.pdf')
    plt.close()

if __name__ == '__main__':
    file1 = 'mistral_expert_alloc.json'
    dataset, all_layerIF, all_mdl, negative_layerIF, negative_mdl = read_mistral_expert_alloc(file1)
    title = 'Expert Allocation for Mistral-7B-v0.1 (5 epochs)'
    plot_mistral(dataset, all_layerIF, all_mdl, negative_layerIF, negative_mdl, title)

    file2 = 'gemma_expert_alloc.json'
    dataset, layerIF, mdl = read_gemma_expert_alloc(file2)
    title = 'Expert Allocation for Gemma-7B (5 epochs)'
    plot_gemma(dataset, layerIF, mdl, title)