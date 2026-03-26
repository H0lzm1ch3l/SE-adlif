import numpy as np
from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd

# ==========================================
# 1. RAW DATA
# ==========================================
shd_lif = [0.883392215, 0.894876301, 0.894876301, 0.908568919, 0.893992961, 0.876325071, 0.884275615, 0.869699657, 0.877650201, 0.880742073]
shd_se_adlif = [0.949646652, 0.931978822, 0.940371037, 0.939045966, 0.933303893, 0.936837435, 0.942579508, 0.944346309, 0.935954034, 0.931095421]
shd_1mclif = [0.895759702, 0.903710246, 0.901060045, 0.912544191, 0.912544191, 0.914310932, 0.898851573, 0.902826846, 0.908568919, 0.881625414]
shd_2mclif = [0.912544191, 0.907243788, 0.901501775, 0.91121906, 0.905918717, 0.907243788, 0.898851573, 0.898851573, 0.89310956, 0.90989399]
shd_3mclif = [0.878533542, 0.907685518, 0.906360447, 0.908568919, 0.883392215, 0.903710246, 0.908568919, 0.905477047, 0.898409903, 0.897968173]
shd_1mcadlif = [0.950088322, 0.952296793, 0.938604236, 0.945229709, 0.945229709, 0.943462908, 0.935954034, 0.934628963, 0.947879851, 0.949204922]
shd_2mcadlif = [0.94611305, 0.935070693, 0.954063594, 0.94611305, 0.933303893, 0.943021178, 0.945229709, 0.947879851, 0.944346309, 0.939045966]
shd_3mcadlif = [0.942579508, 0.94611305, 0.940812707, 0.941254437, 0.954063594, 0.941254437, 0.941696107, 0.936395764, 0.921819806, 0.935070693]

ssc_3mcadlif = [0.789422035, 0.793150842, 0.79398489, 0.802423716, 0.798891187, 0.806201577, 0.8031106, 0.804582477, 0.805171251, 0.801148057]
ssc_2mcadlif = [0.805514693, 0.804680586, 0.804778755, 0.806888402, 0.809635937, 0.801540554, 0.802570879, 0.807280958, 0.80644685, 0.808065951]
ssc_1mcadlif = [0.806299686, 0.805465579, 0.809979379, 0.806495905, 0.803895593, 0.80708468, 0.806250632, 0.802227437, 0.805269361, 0.807967842]
ssc_3mclif = [0.770532846, 0.770091236, 0.767638087, 0.770925343, 0.76498872, 0.762388408, 0.774408817, 0.771563172, 0.751741707, 0.7585124373435974]
ssc_2mclif = [0.764792442, 0.777107239, 0.775733471, 0.768324971, 0.781915426, 0.76616621, 0.7808851, 0.774948478, 0.76219213, 0.773918152]
ssc_1mclif = [0.767490923, 0.774997532, 0.761063695, 0.771366894, 0.764154673, 0.775340974, 0.769747794, 0.7718575, 0.770630956, 0.765037775]
ssc_lif = [0.761456192, 0.763762116, 0.762241185, 0.758365214, 0.753409863, 0.763320565, 0.76273185, 0.763418674, 0.75596112, 0.759101152]
ssc_se_adlif = [0.797860861, 0.799578071, 0.800706506, 0.804778755, 0.796143651, 0.808899999, 0.8076244, 0.801834941, 0.799921513, 0.806545019]

ecg_lif = [0.649811089, 0.635626733, 0.634956181, 0.743061781, 0.63230139, 0.757600546, 0.786623478, 0.646905541, 0.613325238, 0.766104639]
ecg_1mclif = [0.8592844605445862, 0.844162405, 0.851767063, 0.85608995, 0.861666679, 0.862593412, 0.855408549, 0.860620022, 0.859911382, 0.865471721]
ecg_2mclif = [0.8430776, 0.861475885, 0.850753129, 0.850278854, 0.870797694, 0.812501013, 0.845743299, 0.858526707, 0.854814351, 0.850992978]
ecg_3mclif = [0.8365468978881836, 0.848343611, 0.834982336, 0.862435341, 0.840700805, 0.852339447, 0.82780838, 0.855561197, 0.847651303, 0.853729546]
ecg_se_adlif = [0.870372474, 0.882534444, 0.871293783, 0.877442896, 0.870290697, 0.880550146, 0.871964276, 0.872035146, 0.875431359, 0.878211498]
ecg_1mcadlif = [0.870923102, 0.880899012, 0.878429592, 0.889408588, 0.881978393, 0.868279159, 0.86928767, 0.867052615, 0.878587663, 0.88262713]
ecg_2mcadlif = [0.799292445, 0.880250335, 0.885347307, 0.882905126, 0.881073475, 0.88013041, 0.876423478, 0.886862814, 0.875147879, 0.888182044]
ecg_3mcadlif = [0.834268212, 0.831678867, 0.872476697, 0.86253345, 0.874052167, 0.886475742, 0.834028363, 0.835374832, 0.881820321, 0.872967303]

# ==========================================
# 2. ANOVA & TUKEY FUNCTION
# ==========================================
def run_anova_and_tukey(group_dict, dataset_name):
    """
    Runs a One-Way ANOVA and a Tukey HSD post-hoc test.
    group_dict: A dictionary where keys are model names and values are lists of accuracies.
    """
    print(f"\n======================================================================")
    print(f" Analyzing Dataset/Model: {dataset_name}")
    print(f"======================================================================")
    
    # Extract the lists of accuracies and their names
    model_names = list(group_dict.keys())
    accuracy_lists = list(group_dict.values())
    
    # 1. RUN ONE-WAY ANOVA
    # *accuracy_lists unpacks the lists so f_oneway can read them as separate arguments
    f_stat, p_value = stats.f_oneway(*accuracy_lists)
    
    print(f"--- Step 1: One-Way ANOVA ---")
    print(f"F-statistic: {f_stat:.4f}")
    print(f"p-value:     {p_value:.4f}")
    
    # Define our significance level
    alpha = 0.05
    
    if p_value >= alpha:
        print("\nConclusion: NO significant difference found across the models.")
        print("You can stop here. No post-hoc test is needed.")
        return
    else:
        print("\nConclusion: A STATISTICALLY SIGNIFICANT difference exists somewhere among the models.")
        print("Proceeding to Step 2 (Tukey's HSD) to find out exactly WHICH models differ...\n")
        
    # 2. RUN TUKEY'S HSD POST-HOC TEST
    print(f"--- Step 2: Tukey's HSD Post-Hoc Test ---")
    
    # Tukey's test requires us to flatten our data into one long list of values,
    # and create a matching list of labels so it knows which value belongs to which model.
    all_data = []
    all_labels = []
    
    for name, data_list in group_dict.items():
        all_data.extend(data_list)
        all_labels.extend([name] * len(data_list)) # e.g., creates ['LIF', 'LIF', 'LIF'...]
        
    # Run the test
    tukey_results = pairwise_tukeyhsd(endog=all_data, groups=all_labels, alpha=alpha)
    
    # Print the clean summary table
    print(tukey_results.summary())

# ==========================================
# 3. GROUP DATA INTO DICTIONARIES
# ==========================================
shd_lif_models = {
    "0-Comp (LIF)": shd_lif,
    "1-Comp (MC)": shd_1mclif,
    "2-Comp (MC)": shd_2mclif,
    "3-Comp (MC)": shd_3mclif
}

shd_adlif_models = {
    "0-Comp (adLIF)": shd_se_adlif,
    "1-Comp (MC)": shd_1mcadlif,
    "2-Comp (MC)": shd_2mcadlif,
    "3-Comp (MC)": shd_3mcadlif
}

ssc_lif_models = {
    "0-Comp (LIF)": ssc_lif,
    "1-Comp (MC)": ssc_1mclif,
    "2-Comp (MC)": ssc_2mclif,
    "3-Comp (MC)": ssc_3mclif
}

ssc_adlif_models = {
    "0-Comp (MC)": ssc_se_adlif,
    "1-Comp (MC)": ssc_1mcadlif,
    "2-Comp (MC)": ssc_2mcadlif,
    "3-Comp (MC)": ssc_3mcadlif
}

ecg_lif_models = {
    "0-Comp (LIF)": ecg_lif,
    "1-Comp (MC)": ecg_1mclif,
    "2-Comp (MC)": ecg_2mclif,
    "3-Comp (MC)": ecg_3mclif
}

ecg_adlif_models = {
    "0-Comp (adLIF)": ecg_se_adlif,
    "1-Comp (MC)": ecg_1mcadlif,
    "2-Comp (MC)": ecg_2mcadlif,
    "3-Comp (MC)": ecg_3mcadlif
}

# ==========================================
# 4. EXECUTE ALL TESTS IN A LOOP
# ==========================================
all_experiments = [
    (shd_lif_models, "SHD Dataset (LIF Baseline)"),
    (shd_adlif_models, "SHD Dataset (adLIF Baseline)"),
    (ssc_lif_models, "SSC Dataset (LIF Baseline)"),
    (ssc_adlif_models, "SSC Dataset (adLIF Baseline)"),
    (ecg_lif_models, "ECG Dataset (LIF Baseline)"),
    (ecg_adlif_models, "ECG Dataset (adLIF Baseline)")
]

for models, experiment_name in all_experiments:
    run_anova_and_tukey(models, experiment_name)