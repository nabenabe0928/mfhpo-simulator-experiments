#!/bin/bash -l

module load tools/singularity/3.11
cmd="singularity exec mfhpo-simulator.sif python -m src.remove_files"
echo "Remove failed files"
echo $cmd
$cmd

declare -A rsrc
rsrc["bohb"]="walltime=8:00:00"
rsrc["dehb"]="walltime=8:00:00"
rsrc["smac"]="walltime=8:00:00"
rsrc["random"]="walltime=24:00:00"
rsrc["tpe"]="walltime=8:00:00"
rsrc["hyperband"]="walltime=24:00:00"
rsrc["neps"]="walltime=24:00:00"

for n_workers in 1 2 4 8
do
    vars_to_use="-v SEED_START=0,SEED_END=29,N_WORKERS=${n_workers}"
    for opt_name in random tpe hyperband bohb dehb neps smac
    do
        resource="-l nodes=1:ppn=${n_workers},${rsrc[$opt_name]}"
        cmd="msub ${vars_to_use},OPT_NAME=${opt_name} ${resource} scripts/run.moab"
        echo $cmd
        $cmd
    done
done
