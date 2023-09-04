#!/bin/bash -l

module load tools/singularity/3.11
cmd="singularity exec mfhpo-simulator.sif python -m src.remove_files"
echo "Remove failed files"
echo $cmd
$cmd

for n_workers in 2 4 8
do
    memlimit=$(($n_workers * 15))
    resource="-l nodes=1:ppn=${n_workers},mem=${memlimit}gb"
    for dataset_id in 0 1 2
    do
        for seed in `seq 0 29`
        do
            cmd="msub run_dehb_jahs.moab ${resource} -v SEED=${seed},N_WORKERS=${n_workers},DATASET_ID=${dataset_id}"
            echo $cmd
            $cmd
        done
    done
done
