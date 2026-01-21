#!/bin/bash

# experiments=("ECG_MC1adLIF" "ECG_MC2adLIF" "ECG_MC3adLIF" "SHD_MC1adLIF" "SHD_MC2adLIF" "SHD_MC3adLIF" "SSC_MC1adLIF" "SSC_MC2adLIF" "SSC_MC3adLIF")
# experiments=("ECG_M1CLIF" "ECG_M2CLIF" "ECG_M3CLIF" "SHD_M1CLIF" "SHD_M2CLIF" "SHD_M3CLIF" "SSC_M1CLIF" "SSC_M2CLIF" "SSC_M3CLIF" "SHD_M3CLIF2")
# experiments=("SHD_SE_adLIF" "SHD_M3CLIF2_S, SHD_LIF_S" "SHD_M3CLIF2_L", "SSC LIF")
experiments=("SSC_3MCLIF")
runs=5
gpu_count=1  # Number of available GPUs
gpu_offset=2

# use commit id as name
commit_id=$(git rev-parse --short HEAD)
name="${commit_id}_exp"

# no recurrent command
# exp_name=SHD_LIF_2layer_norecurrent l1.use_recurrent=False l2.use_recurrent=False l1.n_neurons=590 l2.input_size=590 l2.n_neurons=590 l_out.input_size=590

for exp in "${experiments[@]}"; do
    for ((i=1; i<=runs; i++)); do
        gpu_id=$(( (i - 1) % gpu_count + gpu_offset ))
        echo "Running $exp - Iteration $i on GPU $gpu_id"
        if [[ $i -lt $runs ]]; then
            uv run run.py experiment=$exp random_seed=$RANDOM device=cuda:$gpu_id exp_name=$name&
        else
            uv run run.py experiment=$exp random_seed=$RANDOM device=cuda:$gpu_id exp_name=$name
        fi
    done
done

wait  # Wait for all background processes to finish
