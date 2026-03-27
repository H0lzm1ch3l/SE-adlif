import numpy as np
from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd

# ==========================================
# 1. RAW DATA (Ablation Study)
# ==========================================
baseline = [0.764792442, 0.777107239, 0.775733471, 0.768324971, 0.781915426, 0.76616621, 0.7808851, 0.774948478, 0.76219213, 0.773918152]
pure_active = [0.7559611, 0.7634187, 0.7129821, 0.7598861, 0.7626337, 0.7564518, 0.7629771, 0.7505642, 0.7522324, 0.7368757]
pure_passive = [0.7662644, 0.7592484, 0.76597, 0.7617506, 0.7686194, 0.7619959, 0.7616524, 0.7612109, 0.7696987, 0.7612599]
no_dend_recur = [0.7616034, 0.7727407, 0.77068, 0.7687175, 0.7625846, 0.7670003, 0.7633696, 0.7725934, 0.7708272, 0.7690609]
trained_d_thr = [0.7671475, 0.77382, 0.7752429, 0.7687666, 0.7699931, 0.7734766, 0.773771, 0.7630262, 0.762094, 0.7750957]
without_proximal = [0.7216171, 0.7111176, 0.754244, 0.6329604, 0.7197037, 0.72461, 0.6155922, 0.1101462, 0.741782, 0.5732999]

# ==========================================
# 2. ANOVA & TUKEY FUNCTION
# ==========================================
def run_anova_and_tukey(group_dict, study_name):
    """
    Runs a One-Way ANOVA and a Tukey HSD post-hoc test.
    """
    print(f"\n" + "="*70)
    print(f" Analyzing Study: {study_name}")
    print("="*70)
    
    model_names = list(group_dict.keys())
    accuracy_lists = list(group_dict.values())
    
    # 1. RUN ONE-WAY ANOVA
    f_stat, p_value = stats.f_oneway(*accuracy_lists)
    
    print(f"--- Step 1: One-Way ANOVA ---")
    print(f"F-statistic: {f_stat:.4f}")
    print(f"p-value:     {p_value:.4f}")
    
    alpha = 0.05
    
    if p_value >= alpha:
        print("\nConclusion: NO significant difference found across the groups.")
        return
    else:
        print("\nConclusion: STATISTICAL SIGNIFICANCE detected.")
        print("Proceeding to Tukey's HSD...\n")
        
    # 2. RUN TUKEY'S HSD POST-HOC TEST
    all_data = []
    all_labels = []
    
    for name, data_list in group_dict.items():
        all_data.extend(data_list)
        all_labels.extend([name] * len(data_list))
        
    tukey_results = pairwise_tukeyhsd(endog=all_data, groups=all_labels, alpha=alpha)
    print(tukey_results.summary())

# ==========================================
# 3. GROUP DATA INTO DICTIONARY
# ==========================================
recurrence_ablation_study = {
    "Baseline": baseline,
    "Pure_Active": pure_active,
    "Pure_Passive": pure_passive,
    "No_Dend_Recur": no_dend_recur,
    "Trained_D_Thr": trained_d_thr,
    # "Without_Proximal": without_proximal
}

# ==========================================
# 4. EXECUTE
# ==========================================
if __name__ == "__main__":
    run_anova_and_tukey(recurrence_ablation_study, "Dendritic Ablation Study")