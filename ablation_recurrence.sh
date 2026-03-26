#!/bin/bash

# ==========================================
# 5-Run Minimal Recurrence Routing Study
# ==========================================

BASE_EXP="SSC_2MCLIF"        # Matches the yaml filename/config
TARGET_PARAMS=1600000        # Triggers dynamic sizing in run.py

runs=10
gpus=(0 2 3 4 5 6 7)             # Corrected: removed '=' and spaces
gpu_count=${#gpus[@]}          # Automatically get the number of available GPUs
delay=5                      # Delay in seconds between experiments
MAX_CONCURRENT_JOBS=8        # Sliding window concurrency limit

commit_id=$(git rev-parse --short HEAD)

# Lock in the structural baseline from the previous ablation study
STRUCTURAL_CONSTANTS="l1.active_dendrite=True l2.active_dendrite=True l1.passive_dendrite=True l2.passive_dendrite=True l1.proximal_dendrite=False l2.proximal_dendrite=False l1.train_d_thr=False l2.train_d_thr=False"

# Names of our 5 experiments
names=(
    "Recur_Baseline"
    "Recur_LSTM_Style"
    "Recur_Self_Vs_Lateral"
    "Recur_Max_Dense"
    "Recur_Pure_Dendritic"
)

# Hydra command-line overrides specifically for the temporal routing
overrides=(
    # 1. Baseline: S2S + D2D (S2D-Self and S2D-Full OFF)
    "l1.use_recurrent=True l2.use_recurrent=True l1.recurrent_dendrite=True l2.recurrent_dendrite=True l1.soma_to_dendrite_recurrence=False l2.soma_to_dendrite_recurrence=False l1.soma_to_dendrite_full_recurrence=False l2.soma_to_dendrite_full_recurrence=False"
    
    # 2. LSTM-Style: S2S + D2D + S2D-Self
    "l1.use_recurrent=True l2.use_recurrent=True l1.recurrent_dendrite=True l2.recurrent_dendrite=True l1.soma_to_dendrite_recurrence=True l2.soma_to_dendrite_recurrence=True l1.soma_to_dendrite_full_recurrence=False l2.soma_to_dendrite_full_recurrence=False"
    
    # 3. Self Vs Lateral: S2S + S2D-Self (D2D OFF)
    "l1.use_recurrent=True l2.use_recurrent=True l1.recurrent_dendrite=False l2.recurrent_dendrite=False l1.soma_to_dendrite_recurrence=True l2.soma_to_dendrite_recurrence=True l1.soma_to_dendrite_full_recurrence=False l2.soma_to_dendrite_full_recurrence=False"
    
    # 4. Max Dense Context: S2S + D2D + S2D-Full
    "l1.use_recurrent=True l2.use_recurrent=True l1.recurrent_dendrite=True l2.recurrent_dendrite=True l1.soma_to_dendrite_recurrence=False l2.soma_to_dendrite_recurrence=False l1.soma_to_dendrite_full_recurrence=True l2.soma_to_dendrite_full_recurrence=True"
    
    # 5. Pure Dendritic: D2D + S2D-Full (S2S OFF)
    "l1.use_recurrent=False l2.use_recurrent=False l1.recurrent_dendrite=True l2.recurrent_dendrite=True l1.soma_to_dendrite_recurrence=False l2.soma_to_dendrite_recurrence=False l1.soma_to_dendrite_full_recurrence=True l2.soma_to_dendrite_full_recurrence=True"
)

for idx in "${!names[@]}"; do
    abl_name="${names[$idx]}"
    abl_override="${overrides[$idx]}"
    
    exp_name="${BASE_EXP}_${abl_name}_${commit_id}"
    
    echo "=================================================="
    echo "Starting Recurrence Ablation: $abl_name"
    echo "=================================================="

    for ((i=1; i<=runs; i++)); do
        gpu_idx=$(( i % gpu_count ))
        gpu_id=${gpus[$gpu_idx]}
        echo "  -> Spawning Iteration $i on GPU $gpu_id"
        
        # Inject the constants AND the overrides into Hydra in the background
        uv run run.py experiment=$BASE_EXP random_seed=$RANDOM device=cuda:$gpu_id exp_name=$exp_name target_params=$TARGET_PARAMS $STRUCTURAL_CONSTANTS $abl_override &
        
        # Sliding Window Concurrency Control
        while (( $(jobs -p | wc -l) >= MAX_CONCURRENT_JOBS )); do
            wait -n
        done
    done
    
    # Ensure all remaining runs in this specific ablation block finish before moving to the next
    sleep $delay
done
wait 
echo "Recurrence ablation study complete!"