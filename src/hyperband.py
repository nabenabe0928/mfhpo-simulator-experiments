from __future__ import annotations

import logging
import os

import numpy as np

from src.utils import get_bench_instance, get_save_dir_name, parse_args, run_bohb


logging.getLogger("hpbandster").setLevel(logging.CRITICAL)


if __name__ == "__main__":
    args = parse_args()
    save_dir_name = get_save_dir_name(args)
    np.random.seed(args.seed)
    obj_func = get_bench_instance(args)
    sampler = "hyperband"

    run_id = f"{sampler}_bench={args.bench_name}_dataset={args.dataset_id}_nworkers={args.n_workers}_seed={args.seed}"
    fidel_key = "epoch" if "epoch" in obj_func.fidel_keys else "z0"
    run_bohb(
        obj_func=obj_func,
        config_space=obj_func.config_space,
        min_fidel=obj_func.min_fidels[fidel_key],
        max_fidel=obj_func.max_fidels[fidel_key],
        fidel_key=fidel_key,
        n_workers=args.n_workers,
        sampler=sampler,
        save_dir_name=os.path.join(sampler, save_dir_name),
        seed=args.seed,
        tmp_dir=args.tmp_dir,
    )
