#!/bin/bash -l

module load tools/singularity/3.11
cmd="singularity exec mfhpo-simulator.sif python -m src.remove_files"
echo "Remove failed files"
echo $cmd
$cmd

submit_moab () {
    cmd=${1}

    echo $cmd
    $cmd
}

for dataset_id in 0 1 2
do
    for seed in `seq 0 29`
    do
        cmd="msub run_neps_jahs.moab -l nodes=1:ppn=8,mem=120gb -v SEED=${seed},N_WORKERS=8,DATASET_ID=${dataset_id}"
        echo $cmd
        $cmd
    done
done

# 0: cifar10, 1: fashion_mnist, 2: colorectal_histology

cmd="msub run_neps_jahs.moab -l nodes=1:ppn=2,mem=30gb -v SEED=3,N_WORKERS=2,DATASET_ID=1"
submit_moab "$cmd"

cmd="msub run_neps_jahs.moab -l nodes=1:ppn=2,mem=30gb -v SEED=9,N_WORKERS=2,DATASET_ID=1"
submit_moab "$cmd"

cmd="msub run_neps_jahs.moab -l nodes=1:ppn=4,mem=60gb -v SEED=3,N_WORKERS=4,DATASET_ID=2"
submit_moab "$cmd"

cmd="msub run_neps_jahs.moab -l nodes=1:ppn=4,mem=60gb -v SEED=11,N_WORKERS=4,DATASET_ID=1"
submit_moab "$cmd"

cmd="msub run_neps_jahs.moab -l nodes=1:ppn=2,mem=30gb -v SEED=22,N_WORKERS=2,DATASET_ID=1"
submit_moab "$cmd"
