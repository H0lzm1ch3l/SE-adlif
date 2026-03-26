#!/bin/bash

# ==========================================
# 5-Run Minimal Ablation Study
# ==========================================

BASE_EXP="SSC_2MCLIF"        # Matches the yaml filename/config
TARGET_PARAMS=1600000        # Triggers the dynamic sizing in run.py

runs=10
gpu_count=5                  # Update this to your total available GPUs
gpu_offset=2
delay=5                      # Delay in seconds between experiments

# NEW COMPUTE PARAMETER
# How many total runs to allow running simultaneously across all GPUs.
# E.g., if MAX_CONCURRENT=8 and gpu_count=4, each GPU handles 2 jobs at once.
MAX_CONCURRENT_JOBS=50        

commit_id=$(git rev-parse --short HEAD)

# Names of our 5 experiments
names=(
    "Pure_Active"
    "Pure_Passive"
    "No_Dend_Recur"
    "Trained_D_Thr"
    "Without_Proximal"
)

# Hydra command-line overrides for Layer 1 and Layer 2
overrides=(
    # 2. Pure Active: Passive OFF
    "l1.active_dendrite=True l2.active_dendrite=True l1.passive_dendrite=False l2.passive_dendrite=False l1.recurrent_dendrite=True l2.recurrent_dendrite=True l1.train_d_thr=False l2.train_d_thr=False l1.proximal_dendrite=True l2.proximal_dendrite=True"
    
    # 3. Pure Passive: Active OFF, Recurrent Dendrite OFF
    "l1.active_dendrite=False l2.active_dendrite=False l1.passive_dendrite=True l2.passive_dendrite=True l1.recurrent_dendrite=False l2.recurrent_dendrite=False l1.train_d_thr=False l2.train_d_thr=False l1.proximal_dendrite=True l2.proximal_dendrite=True"
    
    # 4. No Dendritic Recur: Recurrent Dendrite OFF
    "l1.active_dendrite=True l2.active_dendrite=True l1.passive_dendrite=True l2.passive_dendrite=True l1.recurrent_dendrite=False l2.recurrent_dendrite=False l1.train_d_thr=False l2.train_d_thr=False l1.proximal_dendrite=True l2.proximal_dendrite=True"
    
    # 5. Trained Threshold: Train D Thr ON
    "l1.active_dendrite=True l2.active_dendrite=True l1.passive_dendrite=True l2.passive_dendrite=True l1.recurrent_dendrite=True l2.recurrent_dendrite=True l1.train_d_thr=True l2.train_d_thr=True l1.proximal_dendrite=True l2.proximal_dendrite=True"
    
    # 6. Proximal Dendrite: Proximal ON
    "l1.active_dendrite=True l2.active_dendrite=True l1.passive_dendrite=True l2.passive_dendrite=True l1.recurrent_dendrite=True l2.recurrent_dendrite=True l1.train_d_thr=False l2.train_d_thr=False l1.proximal_dendrite=False l2.proximal_dendrite=False"
)

for idx in "${!names[@]}"; do
    abl_name="${names[$idx]}"
    abl_override="${overrides[$idx]}"
    
    exp_name="${BASE_EXP}_${abl_name}_${commit_id}"
    
    echo "=================================================="
    echo "Starting Ablation: $abl_name"
    echo "=================================================="

    for ((i=1; i<=runs; i++)); do
        gpu_id=$(( (i - 1) % gpu_count + gpu_offset ))
        echo "  -> Spawning Iteration $i on GPU $gpu_id"
        
        # Inject the overrides into Hydra in the background
        uv run run.py experiment=$BASE_EXP random_seed=$RANDOM device=cuda:$gpu_id exp_name=$exp_name target_params=$TARGET_PARAMS $abl_override &
        
        # Sliding Window Concurrency Control
        # If we hit our concurrency limit, wait for at least one job to finish before continuing the loop
        while (( $(jobs -p | wc -l) >= MAX_CONCURRENT_JOBS )); do
            wait -n
        done
    done
    
    # Ensure all remaining runs in this specific ablation block finish before moving to the next ablation
    sleep $delay
done
wait 
echo "Ablation study complete!"