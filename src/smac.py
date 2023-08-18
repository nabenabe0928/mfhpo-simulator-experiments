from __future__ import annotations

from src.utils import get_bench_instance, get_save_dir_name, parse_args, run_smac


if __name__ == "__main__":
    args = parse_args()
    sampler = "smac"
    save_dir_name = get_save_dir_name(opt_name=sampler, args=args)
    load_every_call = True
    bench = get_bench_instance(args, keep_benchdata=False, load_every_call=load_every_call)
    fidel_key = "epoch" if "epoch" in bench.fidel_keys else "z0"
    run_smac(
        obj_func=bench,
        config_space=bench.config_space,
        min_fidel=bench.min_fidels[fidel_key],
        max_fidel=bench.max_fidels[fidel_key],
        fidel_key=fidel_key,
        n_workers=args.n_workers,
        save_dir_name=save_dir_name,
        sampler=sampler,
        load_every_call=load_every_call,
        seed=args.seed,
        tmp_dir=args.tmp_dir,
    )
