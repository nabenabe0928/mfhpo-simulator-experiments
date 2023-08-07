for seed in `0 49`
do
    for n_workers in 1 2 4 8 16
    do
        export SEED_START=${seed}
        export SEED_END=${seed}
        export N_WORKERS=${n_workers}
        msub run.moab
    done
done
