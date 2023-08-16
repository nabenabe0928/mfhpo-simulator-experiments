from __future__ import annotations

import os
from typing import Any

import ConfigSpace as CS

from benchmark_simulator import ObjectiveFuncWrapper

from dehb import DEHB

import numpy as np

from src.utils import get_bench_instance, get_save_dir_name, parse_args


class DEHBObjectiveFuncWrapper(ObjectiveFuncWrapper):
    # Adapt to the DEHB interface at https://github.com/automl/DEHB/
    def __call__(self, config: CS.Configuration, budget: int, **data_to_scatter: Any) -> dict[str, float]:
        eval_config = config.get_dictionary()
        fidels = {self.fidel_keys[0]: int(budget)}
        results = super().__call__(eval_config=eval_config, fidels=fidels, **data_to_scatter)
        return dict(fitness=results[self.obj_keys[0]], cost=results[self.runtime_key])


def run_dehb(
    obj_func: Any,
    config_space: CS.ConfigurationSpace,
    save_dir_name: str,
    min_fidel: int,
    max_fidel: int,
    fidel_key: str,
    n_workers: int,
    seed: int,
    load_every_call: bool,
    tmp_dir: str | None,
    n_evals: int = 450,  # eta=3,S=2,100 full evals
) -> None:
    data_to_scatter = {}
    if not load_every_call and hasattr(obj_func, "get_benchdata"):
        # This data is shared in memory, and thus the optimization becomes quicker!
        data_to_scatter = {"benchdata": obj_func.get_benchdata()}

    np.random.seed(seed)
    n_actual_evals_in_opt = n_evals + n_workers
    wrapper = DEHBObjectiveFuncWrapper(
        save_dir_name=save_dir_name,
        n_workers=n_workers,
        obj_func=obj_func,
        n_actual_evals_in_opt=n_actual_evals_in_opt,
        n_evals=n_evals,
        continual_max_fidel=max_fidel,
        fidel_keys=[fidel_key],
        max_waiting_time=120.0,
        seed=seed,
        tmp_dir=tmp_dir,
    )

    dehb = DEHB(
        f=wrapper,
        cs=config_space,
        min_budget=min_fidel,
        max_budget=max_fidel,
        n_workers=n_workers,
        output_path=os.path.join("" if tmp_dir is None else tmp_dir, "logs/dehb-log"),
    )

    dehb.run(fevals=n_actual_evals_in_opt, **data_to_scatter)


if __name__ == "__main__":
    args = parse_args()
    save_dir_name = get_save_dir_name(opt_name="dehb", args=args)
    load_every_call = bool(args.n_workers != 1)
    bench = get_bench_instance(args, keep_benchdata=False, load_every_call=load_every_call)
    fidel_key = "epoch" if "epoch" in bench.fidel_keys else "z0"
    run_dehb(
        obj_func=bench,
        config_space=bench.config_space,
        min_fidel=bench.min_fidels[fidel_key],
        max_fidel=bench.max_fidels[fidel_key],
        fidel_key=fidel_key,
        n_workers=args.n_workers,
        load_every_call=load_every_call,
        save_dir_name=save_dir_name,
        seed=args.seed,
        tmp_dir=args.tmp_dir,
    )
