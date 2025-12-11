#!/bin/bash

# experiments=("ECG_MC1adLIF" "ECG_MC2adLIF" "ECG_MC3adLIF" "SHD_MC1adLIF" "SHD_MC2adLIF" "SHD_MC3adLIF" "SSC_MC1adLIF" "SSC_MC2adLIF" "SSC_MC3adLIF")
# experiments=("ECG_M1CLIF" "ECG_M2CLIF" "ECG_M3CLIF" "SHD_M1CLIF" "SHD_M2CLIF" "SHD_M3CLIF" "SSC_M1CLIF" "SSC_M2CLIF" "SSC_M3CLIF")
experiments=("SSC_MC1adLIF" "SSC_MC2adLIF" "SSC_MC3adLIF")
runs=5
gpu_count=3  # Number of available GPUs
gpu_offset=4

for exp in "${experiments[@]}"; do
    for ((i=1; i<=runs; i++)); do
        gpu_id=$(( ((i - 1) + gpu_offset) % gpu_count ))
        echo "Running $exp - Iteration $i on GPU $gpu_id"
        if [[ $i -lt $runs ]]; then
            uv run run.py experiment=$exp device=cuda:$gpu_id &
        else
            uv run run.py experiment=$exp device=cuda:$gpu_id
        fi
    done
done

wait  # Wait for all background processes to finish
