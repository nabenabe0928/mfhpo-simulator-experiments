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
    name_to_id["colorectal-histology"]=0

    memlimit=$(($n_workers * 15))
    cmd="msub run_neps_jahs.moab -l nodes=1:ppn=${n_workers},mem=${memlimit}gb -v SEED=${seed},N_WORKERS=${n_workers},DATASET_ID=${name_to_id[$dataset_name]}"
    echo $cmd
    # $cmd
}

for seed in 3 10 14 25
do
    submit_moab $seed "colorectal-histology" 4
done

for seed in 5 9 10 11 14 17 18 21 22 24 28
do
    submit_moab $seed "fashion-mnist" 8
done

for seed in 11 18
do
    submit_moab $seed "fashion-mnist" 4
done

for seed in 2 4 7 8 10 11 12 21 29
do
    submit_moab $seed "colorectal-histology" 8
done

for seed in 3 9 22
do
    submit_moab $seed "fashion-mnist" 2
done
