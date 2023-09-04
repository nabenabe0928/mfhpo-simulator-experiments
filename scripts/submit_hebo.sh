#!/bin/bash -l

module load tools/singularity/3.11
cmd="singularity exec mfhpo-simulator.sif python -m src.remove_files"
echo "Remove failed files"
echo $cmd
$cmd

submit_light_bench () {
    fixed_vars_to_use=${1}

    declare -A bench_max_id
    bench_max_id["lc"]=33
    bench_max_id["hpobench"]=7

    for bench_name in "hpobench" "lc"
    do
        for dataset_id in `seq 0 ${bench_max_id[$bench_name]}`
        do
            vars_to_use="${fixed_vars_to_use},BENCH_NAME=${bench_name},DATASET_ID=${dataset_id}"
            cmd="msub ${vars_to_use} scripts/run_hebo.moab"
            echo $cmd
            $cmd
        done
    done

    for other_vars_to_use in "BENCH_NAME=hartmann,DIM=3" "BENCH_NAME=hartmann,DIM=6" "BENCH_NAME=branin"
    do
        vars_to_use="${fixed_vars_to_use},${other_vars_to_use}"
        cmd="msub ${vars_to_use} scripts/run_hebo.moab"
        echo $cmd
        $cmd
    done
}

submit_heavy_bench () {
    fixed_vars_to_use=${1}

    declare -A bench_max_id
    bench_max_id["hpolib"]=3
    bench_max_id["jahs"]=2

    for bench_name in "hpolib" "jahs"
    do
        for dataset_id in `seq 0 ${bench_max_id[$bench_name]}`
        do
            vars_to_use="${fixed_vars_to_use},BENCH_NAME=${bench_name},DATASET_ID=${dataset_id}"
            cmd="msub ${vars_to_use} scripts/run_hebo.moab"
            echo $cmd
            $cmd
        done
    done
}

for s in 0 1 2
do
    for n_workers in 1 2 4 8
    do
        seed_start=$(($s * 10))
        seed_end=$(($seed_start + 9))
        fixed_vars_to_use="-v SEED_START=${seed_start},SEED_END=${seed_end},N_WORKERS=${n_workers}"
        submit_heavy_bench "$fixed_vars_to_use"
        if [[ "$s" == 0 ]]
        then
            fixed_vars_to_use="-v SEED_START=0,SEED_END=29,N_WORKERS=${n_workers}"
            submit_light_bench "$fixed_vars_to_use"
        fi
    done
done
