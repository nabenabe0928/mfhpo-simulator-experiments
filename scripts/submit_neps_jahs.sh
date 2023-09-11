#!/bin/bash -l

module load tools/singularity/3.11
cmd="singularity exec mfhpo-simulator.sif python -m src.remove_files"
echo "Remove failed files"
echo $cmd
$cmd

submit_moab () {
    seed=${1}
    dataset_name=${2}
    n_workers=${3}

    declare -A name_to_id
    name_to_id["cifar10"]=0
    name_to_id["fashion-mnist"]=1
    name_to_id["colorectal-histology"]=2

    memlimit=$(($n_workers * 15))
    cmd="msub scripts/run_neps_jahs.moab -l nodes=1:ppn=${n_workers},mem=${memlimit}gb -v SEED=${seed},N_WORKERS=${n_workers},DATASET_ID=${name_to_id[$dataset_name]}"
    echo $cmd
    $cmd
}

for seed in `seq 0 29`
do
    for n_workers in 2 4 8
    do
        for dataset_name in cifar10 fashion-mnist colorectal-histology
        do
            submit_moab $seed $dataset_name $n_workers
        done
    done
done
