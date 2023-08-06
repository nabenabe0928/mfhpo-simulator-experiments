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

for seed in `seq 0 9`
do
    for n_workers in 1 2 4 8 16 32 64
    do
        subcmd="python -m src.smac --seed ${seed} --n_workers ${n_workers}"
        run_bench "$subcmd"
    done
done
