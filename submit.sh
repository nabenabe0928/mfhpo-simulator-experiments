#!/bin/bash -l

module load tools/singularity/3.11
cmd="singularity exec mfhpo-simulator.sif python -m src.remove_files"
echo "Remove failed files"
echo $cmd
$cmd

declare -A rsrc
rsrc["bohb"]="walltime=8:00:00"
rsrc["dehb"]="walltime=2:00:00"
rsrc["smac"]="walltime=8:00:00"
rsrc["random"]="walltime=24:00:00"
rsrc["tpe"]="walltime=8:00:00"
rsrc["hyperband"]="walltime=24:00:00"
rsrc["neps"]="walltime=24:00:00"

for seed in `seq 0 29`
do
    for n_workers in 1 2 4 8
    do
        vars_to_use="-v SEED_START=${seed},SEED_END=${seed},N_WORKERS=${n_workers}"
        for opt_name in random tpe hyperband bohb dehb neps smac
        do
            memlimit=$(($n_workers * 15))
            resource="-l nodes=1:ppn=${n_workers},${rsrc[$opt_name]},mem=${memlimit}gb"
            cmd="msub ${vars_to_use},OPT_NAME=${opt_name} ${resource} scripts/run.moab"
            echo $cmd
            $cmd
        done
    done
done
