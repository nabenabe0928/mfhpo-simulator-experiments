#!/bin/bash -l

for seed in `seq 0 29`
do
    for n_workers in 1 2 4 8
    do
        vars_to_use="-v SEED_START=${seed},SEED_END=${seed},N_WORKERS=${n_workers}"
        resource="-l nodes=1:ppn=${n_workers}"

        cmd="msub ${vars_to_use} ${resource} scripts/run.moab"
        echo $cmd
        $cmd
        cmd="msub ${vars_to_use} scripts/run_hebo.moab"
        echo $cmd
        $cmd
    done
done
