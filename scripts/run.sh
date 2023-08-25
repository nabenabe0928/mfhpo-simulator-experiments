#!/bin/bash -l

while [[ $# -gt 0 ]]; do
    key="$1"

    case $key in
        --opt_name)
            opt_name="$2"
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
        --bench_name)
            bench_name="$2"
            shift
            shift
            ;;
        --dataset_id)
            dataset_id="$2"
            shift
            shift
            ;;
        --dim)
            dim="$2"
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

    if [[ "$bench_name" == "branin" ]]
    then
        cmd="${subcmd}"
    elif [[ "$bench_name" == "hartmann" ]]
    then
        cmd="${subcmd} --dim ${dim}"
    else
        cmd="${subcmd} --dataset_id ${dataset_id}"
    fi

    echo `date '+%y/%m/%d %H:%M:%S'`
    echo $cmd
    $cmd
}

declare -A exec_cmds
exec_cmds["bohb"]="python -m src.bohb"
exec_cmds["dehb"]="python -m src.dehb"
exec_cmds["smac"]="python -m src.smac"
exec_cmds["random"]="python -m src.random"
exec_cmds["tpe"]="python -m src.tpe"
exec_cmds["hyperband"]="python -m src.hyperband"
exec_cmds["neps"]="./src/neps.sh"

exec_cmd=${exec_cmds[$opt_name]}
fixed_cmd="${exec_cmd} --n_workers ${n_workers} --tmp_dir ${TMPDIR} --bench_name ${bench_name}"
for seed in `seq ${seed_start} ${seed_end}`
do
    subcmd="${fixed_cmd} --seed ${seed}"
    run_bench "$subcmd"
done

echo "Finished run.sh with opt_name=${opt_name}!!"
