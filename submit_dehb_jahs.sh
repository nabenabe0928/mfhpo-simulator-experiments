#!/bin/bash -l

module load tools/singularity/3.11
cmd="singularity exec mfhpo-simulator.sif python -m src.remove_files"
echo "Remove failed files"
echo $cmd
$cmd

for dataset_id in 1 2
do
    for seed in `seq 0 29`
    do
        cmd="msub run_dehb_jahs.moab -v SEED=${seed},DATASET_ID=${dataset_id}"
        echo $cmd
        $cmd
    done
done
