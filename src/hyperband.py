from __future__ import annotations

import os

from src.utils import get_bench_instance, get_save_dir_name, parse_args, run_smac


if __name__ == "__main__":
    args = parse_args()
    save_dir_name = get_save_dir_name(args)
    bench = get_bench_instance(args, keep_benchdata=False)
    fidel_key = "epoch" if "epoch" in bench.fidel_keys else "z0"
    sampler = "hyperband"
    run_smac(
        obj_func=bench,
        config_space=bench.config_space,
        min_fidel=bench.min_fidels[fidel_key],
        max_fidel=bench.max_fidels[fidel_key],
        fidel_key=fidel_key,
        n_workers=args.n_workers,
        save_dir_name=os.path.join(sampler, save_dir_name),
        sampler=sampler,
        seed=args.seed,
        tmp_dir=args.tmp_dir,
    )
