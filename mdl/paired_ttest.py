import scipy.stats as stats
import numpy as np

# Data from Table 4: Mistral-7B-v0.1 (+ve variant)
# Datasets: CoLA, MRPC, CommonsenseQA, OpenBookQA, ScienceQA
layerif_mistral_ve = np.array([86.10, 82.84, 81.40, 85.80, 80.80])
mdl_mistral_ve = np.array([85.90, 84.23, 80.18, 86.40, 83.59])

# Data from Table 4: Mistral-7B-v0.1 (All variant)
layerif_mistral_all = np.array([85.62, 83.59, 81.24, 85.20, 66.41])
mdl_mistral_all = np.array([87.44, 82.72, 80.43, 85.00, 79.77])

# Data from Table 5: Gemma-7B (+ve variant)
layerif_gemma = np.array([87.15, 84.41, 82.64, 88.40, 94.69])
mdl_gemma = np.array([87.34, 84.99, 81.57, 88.80, 94.92])

def run_paired_ttest(name, mdl_data, layerif_data):
    # We use alternative='greater' because the hypothesis is that 
    # the MDL framework improves performance over the heuristic baseline.
    res = stats.ttest_rel(mdl_data, layerif_data, alternative='greater')
    
    mean_diff = np.mean(mdl_data) - np.mean(layerif_data)
    
    print(f"--- {name} ---")
    print(f"Mean Difference (MDL - LayerIF): {mean_diff:.2f}%")
    print(f"MDL Mean: {np.mean(mdl_data):.2f}%, LayerIF Mean: {np.mean(layerif_data):.2f}%")
    print(f"T-statistic: {res.statistic:.4f}")
    print(f"P-value: {res.pvalue:.4f}")
    
    if res.pvalue < 0.05:
        print("Result: Statistically Significant (p < 0.05)\n")
    else:
        print("Result: Not Statistically Significant (p >= 0.05)\n")

# Run tests for individual configurations
run_paired_ttest("Mistral-7B (+ve)", mdl_mistral_ve, layerif_mistral_ve)
run_paired_ttest("Mistral-7B (All)", mdl_mistral_all, layerif_mistral_all)
run_paired_ttest("Gemma-7B (+ve)", mdl_gemma, layerif_gemma)

# Run combined test for maximum statistical power
print("======================================================")
all_layerif = np.concatenate([layerif_mistral_ve, layerif_mistral_all, layerif_gemma])
all_mdl = np.concatenate([mdl_mistral_ve, mdl_mistral_all, mdl_gemma])
run_paired_ttest("Combined Across All Models & Variants (N=15)", all_mdl, all_layerif)