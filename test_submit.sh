#!/bin/bash -l

for seed in 0
do
    for n_workers in 4 8
    do
        vars_to_use="-v SEED_START=${seed},SEED_END=${seed},N_WORKERS=${n_workers}"
        resource="-l nodes=1:ppn=${n_workers}"

        cmd="msub ${vars_to_use} ${resource} scripts/test_run.moab"
        echo $cmd
        $cmd
    done
done
