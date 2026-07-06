#!/bin/bash

# ==========================================
# DH-SRNN Architecture Ablation Study
# ==========================================
# Tests: Bidirectional vs Unidirectional + Sparse vs Dense connections
# 
# Parameter behavior:
# - Bidirectionality: COMPENSATED via 1.5x factor → always 1.6M params
# - Sparse: NOT compensated → results in FEWER actual params
# - Target params (1.6M) applies to non-sparse configs; sparse configs are lower

BASE_EXP="SSC_DH_SRNN"       # Matches the yaml filename/config
TARGET_PARAMS=1600000        # Target parameter count (1.6M)

runs=10
gpu_count=5                  # Update this to your total available GPUs
gpu_offset=2
delay=5                      # Delay in seconds between experiments

# NEW COMPUTE PARAMETER
# How many total runs to allow running simultaneously across all GPUs.
MAX_CONCURRENT_JOBS=50        

commit_id=$(git rev-parse --short HEAD)

# Names of our 4 experiments (2x2 ablation: bidirectional × sparse)
names=(
    "Uni_Dense"           # Unidirectional, No sparse (baseline)
    "Bi_Dense"            # Bidirectional, No sparse
    "Uni_Sparse"          # Unidirectional, With sparse
    "Bi_Sparse"           # Bidirectional, With sparse
)

# Hydra command-line overrides
# Each experiment keeps the same target_params but varies architecture
overrides=(
    # 1. Uni_Dense: No bidirectional, No sparse
    "use_bidirectional=false use_sparse=false"
    
    # 2. Bi_Dense: Bidirectional, No sparse
    "use_bidirectional=true use_sparse=false"
    
    # 3. Uni_Sparse: No bidirectional, With sparse
    "use_bidirectional=false use_sparse=true"
    
    # 4. Bi_Sparse: Bidirectional, With sparse
    "use_bidirectional=true use_sparse=true"
)

for idx in "${!names[@]}"; do
    abl_name="${names[$idx]}"
    abl_override="${overrides[$idx]}"
    
    exp_name="${BASE_EXP}_${abl_name}_${commit_id}"
    
    echo "=================================================="
    echo "Starting Ablation: $abl_name"
    echo "  - Bidirectional: $(echo $abl_override | grep -o 'use_bidirectional=[^ ]*' | cut -d= -f2)"
    echo "  - Sparse: $(echo $abl_override | grep -o 'use_sparse=[^ ]*' | cut -d= -f2)"
    echo "  - Target Params: $TARGET_PARAMS"
    echo "=================================================="

    for ((i=1; i<=runs; i++)); do
        gpu_id=$(( (i - 1) % gpu_count + gpu_offset ))
        echo "  -> Spawning Iteration $i on GPU $gpu_id"
        
        # Inject the overrides into Hydra in the background
        # Note: target_params stays constant - sparse configs will get more neurons
        # to compensate for the reduced parameter count from sparsity
        uv run run.py experiment=$BASE_EXP random_seed=$RANDOM device=cuda:$gpu_id exp_name=$exp_name target_params=$TARGET_PARAMS $abl_override &
        
        # Sliding Window Concurrency Control
        # If we hit our concurrency limit, wait for at least one job to finish
        while (( $(jobs -p | wc -l) >= MAX_CONCURRENT_JOBS )); do
            wait -n
        done
    done
    
    # Ensure all remaining runs in this specific ablation block finish
    sleep $delay
done
wait 
echo "Ablation study complete!"
echo ""
echo "Summary of configurations tested:"
echo "  1. Uni_Dense    - Unidirectional, Dense connections (baseline)"
echo "  2. Bi_Dense     - Bidirectional, Dense connections (1.5x factor)"
echo "  3. Uni_Sparse   - Unidirectional, Sparse connections (reduced params)"
echo "  4. Bi_Sparse    - Bidirectional, Sparse connections"
echo ""
echo "Note: Parameter behavior:"
echo "      - Bidirectionality: COMPENSATED (1.5x factor) → 1.6M params"
echo "      - Sparse: NOT compensated → actual params < 1.6M"