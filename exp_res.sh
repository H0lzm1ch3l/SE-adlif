#!/bin/bash

experiments=("SSC_M1CLIF" "SSC_M2CLIF" "SSC_M3CLIF")
runs=5
gpu_count=8  # Number of available GPUs
gpu_offset=0

for exp in "${experiments[@]}"; do
    for ((i=1; i<=runs; i++)); do
        gpu_id=$(( (i - 1 + gpu_offset) % gpu_count ))
        echo "Running $exp - Iteration $i on GPU $gpu_id"
        if [[ $i -lt $runs ]]; then
            uv run run.py experiment=$exp device=cuda:$gpu_id &
        else
            uv run run.py experiment=$exp device=cuda:$gpu_id
        fi
    done
done

wait  # Wait for all background processes to finish
