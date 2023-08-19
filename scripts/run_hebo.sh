#!/bin/bash -l

while [[ $# -gt 0 ]]; do
    key="$1"

    case $key in
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

prefix="python -m src.hebo --bench_name ${bench_name} --n_workers ${n_workers}"
if [[ "$bench_name" == "hartmann" ]]
then
    suffix="--dim ${dim}"
elif [[ "$bench_name" == "branin" ]]
then
    suffix=""
else
    suffix="--dataset_id ${dataset_id}"
fi

for seed in `seq ${seed_start} ${seed_end}`
do
    echo `date '+%y/%m/%d %H:%M:%S'`
    cmd="${prefix} ${suffix} --seed ${seed}"
    echo $cmd
    $cmd
done

echo "Finished run.sh for HEBO!!"
