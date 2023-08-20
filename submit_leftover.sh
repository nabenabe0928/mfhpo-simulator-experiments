#!/bin/bash -l

module load tools/singularity/3.11
cmd="singularity exec mfhpo-simulator.sif python -m src.remove_files"
echo "Remove failed files"
echo $cmd
$cmd


for n_workers in 1 2 4 8
do
    vars_to_use="-v SEED_START=0,SEED_END=29,N_WORKERS=${n_workers}"
    resource="-l nodes=1:ppn=${n_workers}"

    for mode in hb oss neps
    do
        cmd="msub ${vars_to_use},MODE=${mode} ${resource} scripts/run.moab"
        echo $cmd
        $cmd
    done
done
