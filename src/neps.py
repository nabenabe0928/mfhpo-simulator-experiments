from __future__ import annotations

import os
import warnings
from typing import Any

import ConfigSpace as CS

from benchmark_simulator import ObjectiveFuncWrapper

import neps

import numpy as np

from src.utils import get_bench_instance, get_save_dir_name, parse_args


warnings.filterwarnings("ignore", category=DeprecationWarning)


class NEPSWorker(ObjectiveFuncWrapper):
    def __call__(self, **eval_config: dict[str, Any]) -> dict[str, float]:
        _eval_config = eval_config.copy()
        fidel_key = self.fidel_keys[0]
        fidels = {fidel_key: _eval_config.pop(fidel_key)}
        return super().__call__(eval_config=_eval_config, fidels=fidels)


def get_pipeline_space(config_space: CS.ConfigurationSpace) -> dict[str, neps.search_spaces.parameter.Parameter]:
    pipeline_space = {}
    for hp_name in config_space:
        hp = config_space.get_hyperparameter(hp_name)
        if isinstance(hp, CS.UniformFloatHyperparameter):
            pipeline_space[hp.name] = neps.FloatParameter(lower=hp.lower, upper=hp.upper, log=hp.log)
        elif isinstance(hp, CS.UniformIntegerHyperparameter):
            pipeline_space[hp.name] = neps.IntegerParameter(lower=hp.lower, upper=hp.upper, log=hp.log)
        elif isinstance(hp, CS.CategoricalHyperparameter):
            pipeline_space[hp.name] = neps.CategoricalParameter(choices=hp.choices)
        else:
            raise ValueError(f"{type(hp)} is not supported")

    return pipeline_space


def run_neps(
    obj_func: Any,
    config_space: CS.ConfigurationSpace,
    save_dir_name: str,
    min_fidel: int,
    max_fidel: int,
    fidel_key: str,
    worker_index: int,
    n_workers: int,
    seed: int,
    tmp_dir: str | None,
    n_evals: int = 450,  # eta=3,S=2,100 full evals
):
    np.random.seed(seed)
    n_actual_evals_in_opt = n_evals + n_workers
    worker = NEPSWorker(
        save_dir_name=save_dir_name,
        launch_multiple_wrappers_from_user_side=True,
        n_workers=n_workers,
        obj_func=obj_func,
        n_actual_evals_in_opt=n_actual_evals_in_opt,
        n_evals=n_evals,
        fidel_keys=[fidel_key],
        continual_max_fidel=max_fidel,
        seed=seed,
        max_waiting_time=30.0,
        tmp_dir=tmp_dir,
        worker_index=worker_index,
    )
    pipeline_space = get_pipeline_space(config_space)
    pipeline_space[fidel_key] = neps.IntegerParameter(lower=min_fidel, upper=max_fidel, is_fidelity=True)

    neps.run(
        run_pipeline=worker,
        pipeline_space=pipeline_space,
        root_directory=os.path.join("" if tmp_dir is None else tmp_dir, "logs", "_".join(save_dir_name.split("/"))),
        max_evaluations_total=n_actual_evals_in_opt,
    )


if __name__ == "__main__":
    args = parse_args()
    save_dir_name = get_save_dir_name(args)
    bench = get_bench_instance(args)
    fidel_key = "epoch" if "epoch" in bench.fidel_keys else "z0"

    run_neps(
        obj_func=bench,
        config_space=bench.config_space,
        min_fidel=bench.min_fidels[fidel_key],
        max_fidel=bench.max_fidels[fidel_key],
        fidel_key=fidel_key,
        n_workers=args.n_workers,
        save_dir_name=os.path.join("neps", save_dir_name),
        seed=args.seed,
        tmp_dir=args.tmp_dir,
        worker_index=args.worker_index,
    )
