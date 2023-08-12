#!/bin/bash -l

for seed in `seq 0 2`
do
    for n_workers in 4 8
    do
        export SEED_START=${seed}
        export SEED_END=${seed}
        export N_WORKERS=${n_workers}
        msub test_run.moab
    done
done
