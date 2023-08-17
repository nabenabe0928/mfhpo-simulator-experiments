#!/bin/bash -l

while [[ $# -gt 0 ]]; do
    key="$1"

    case $key in
        --mode)
            mode="$2"
            shift
            shift
            ;;
        --seed_start)
            seed_start="$2"
            shift
            shift
            ;;
        --seed_end)
            seed_end="$2"
            shift
            shift
            ;;
        --n_workers)
            n_workers="$2"
            shift
            shift
            ;;
        *)
            shift
            ;;
    esac
done

run_bench () {
    subcmd=${1}

    declare -A bench_max_id
    bench_max_id["hpolib"]=3
    bench_max_id["lc"]=33
    bench_max_id["jahs"]=2
    bench_max_id["hpobench"]=7

    for bench_name in hpolib hpobench lc jahs
    do
        for dataset_id in `seq 0 ${bench_max_id[$bench_name]}`
        do
            cmd="${subcmd} --bench_name ${bench_name} --dataset_id ${dataset_id}"
            echo `date '+%y/%m/%d %H:%M:%S'`
            echo $cmd
            $cmd
        done
    done

    for subcmd2 in "--bench_name hartmann --dim 3" "--bench_name hartmann --dim 6" "--bench_name branin"
    do
        cmd="${subcmd} ${subcmd2}"
        echo `date '+%y/%m/%d %H:%M:%S'`
        echo $cmd
        $cmd
    done
}

run_opt () {
    opt_name=${1}

    declare -A exec_cmds
    exec_cmds["bohb"]="python -m src.bohb"
    exec_cmds["dehb"]="python -m src.dehb"
    exec_cmds["smac"]="python -m src.smac"
    exec_cmds["random"]="python -m src.random"
    exec_cmds["tpe"]="python -m src.tpe"
    exec_cmds["hyperband"]="python -m src.hyperband"
    exec_cmds["neps"]="./src/neps.sh"

    exec_cmd=${exec_cmds[$opt_name]}
    for seed in `seq ${seed_start} ${seed_end}`
    do
        subcmd="${exec_cmd} --seed ${seed} --n_workers ${n_workers} --tmp_dir ${TMPDIR}"
        run_bench "$subcmd"
    done

    echo "Start the result synchronization for ${opt_name}"
    rsync -a $TMPDIR/mfhpo-simulator-info/ mfhpo-simulator-info/
}

if [[ "$mode" == "smac" ]]
then
    run_opt "smac"
elif [[ "$mode" == "hb" ]]
then
    run_opt "bohb"
    run_opt "dehb"
    run_opt "hyperband"
elif [[ "$mode" == "optuna" ]]
then
    run_opt "random"
    run_opt "tpe"
else
    run_opt "neps"
fi

echo `date '+%y/%m/%d %H:%M:%S'`
echo "Finished run.sh with mode=${mode}!!"
