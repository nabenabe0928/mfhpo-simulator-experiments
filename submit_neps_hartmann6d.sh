#!/bin/bash -l

module load tools/singularity/3.11
cmd="singularity exec mfhpo-simulator.sif python -m src.remove_files"
echo "Remove failed files"
echo $cmd
$cmd

for seed in 0 10 20
do
    for n_workers in 1 2 4 8
    do
        memlimit=$(($n_workers * 4))
        cmd="msub run_neps_hartmann6d.moab -l nodes=1:ppn=${n_workers},mem=${memlimit}gb -v SEED=${seed},N_WORKERS=${n_workers}"
        echo $cmd
        $cmd
    done
done
