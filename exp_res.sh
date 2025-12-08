#!/bin/bash

experiments=("ECG_M1CLIF" "ECG_M2CLIF" "ECG_M3CLIF" "SHD_M1CLIF" "SHD_M2CLIF" "SHD_M3CLIF" "SSC_M1CLIF" "SSC_M2CLIF" "SSC_M3CLIF")
runs=5
gpu_count=8  # Number of available GPUs

for exp in "${experiments[@]}"; do
    for ((i=1; i<=runs; i++)); do
        gpu_id=$(( (i - 1) % gpu_count ))
        echo "Running $exp - Iteration $i on GPU $gpu_id"
        uv run run.py experiment=$exp device=cuda:$gpu_id &
    done
done

wait  # Wait for all background processes to finish
