#!/bin/bash

# experiments=("ECG_MC1adLIF" "ECG_MC2adLIF" "ECG_MC3adLIF" "SHD_MC1adLIF" "SHD_MC2adLIF" "SHD_MC3adLIF" "SSC_MC1adLIF" "SSC_MC2adLIF" "SSC_MC3adLIF")
# experiments=("ECG_M1CLIF" "ECG_M2CLIF" "ECG_M3CLIF" "SHD_M1CLIF" "SHD_M2CLIF" "SHD_M3CLIF" "SSC_M1CLIF" "SSC_M2CLIF" "SSC_M3CLIF" "SHD_M3CLIF2")
# experiments=("SHD_SE_adLIF" "SHD_M3CLIF2_S, SHD_LIF_S" "SHD_M3CLIF2_L", "SSC LIF")
# experiments=("SSC_3MCLIF" "SSC_2MCLIF" "SSC_1MCLIF")
# experiments=("SSC_1MCadLIF") #  "SSC_2MCadLIF") #  "SSC_3MCadLIF")
# experiments=("ECG_LIF" "ECG_1MCLIF" "ECG_2MCLIF" "ECG_3MCLIF" "ECG_1MCadLIF" "ECG_2MCadLIF" "ECG_3MCadLIF")
# experiments=("SSC_1MCLIF")
# experiments=("SSC_1MCadLIF")
# experiments=("SSC_SE_adLIF")
# experiments=("SSC_adLIF")
# experiments=("SHD_LIF" "SHD_SE_adLIF")
# experiments=("SHD_3MCadLIF" "SHD_2MCadLIF" "SHD_1MCadLIF")
# experiments=("ECG_LIF" "ECG_1MCLIF" "ECG_2MCLIF" "ECG_3MCLIF" "ECG_1MCadLIF" "ECG_2MCadLIF" "ECG_3MCadLIF" "ECG_SE_adLIF_2layer")
# experiments=("SHD_3MCadLIF" "SHD_2MCadLIF" "SHD_1MCadLIF" "SHD_3MCLIF" "SHD_2MCLIF")
experiments=("SHD_SE_adLIF")
runs=10
gpu_count=1
gpu_offset=2
delay=5  # Delay in seconds between experiments

commit_id=$(git rev-parse --short HEAD)

for exp in "${experiments[@]}"; do
    name="${exp}_${commit_id}"
    for ((i=1; i<=runs; i++)); do
        gpu_id=$(( (i - 1) % gpu_count + gpu_offset ))
        echo "Running $exp - Iteration $i on GPU $gpu_id"
        if [[ $i -lt $runs || $exp != ${experiments[-1]} ]]; then
            uv run run.py experiment=$exp random_seed=$RANDOM device=cuda:$gpu_id exp_name=$name &
        else
            uv run run.py experiment=$exp random_seed=$RANDOM device=cuda:$gpu_id exp_name=$name
        fi
    done
    [[ $exp != ${experiments[-1]} ]] && sleep $delay
done
wait