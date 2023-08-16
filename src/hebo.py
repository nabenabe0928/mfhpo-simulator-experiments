from __future__ import annotations

import time
from typing import Any

import ConfigSpace as CS

from benchmark_simulator import AbstractAskTellOptimizer, ObjectiveFuncWrapper

from hebo.design_space.design_space import DesignSpace
from hebo.optimizers.hebo import HEBO

import numpy as np

import pandas as pd

from src.utils import get_bench_instance, get_save_dir_name, parse_args


def extract_space(config_space: CS.ConfigurationSpace):
    config_info = []
    for hp in config_space.get_hyperparameters():
        info = {"name": hp.name}
        if isinstance(hp, CS.CategoricalHyperparameter):
            info["type"] = "cat"
            info["categories"] = hp.choices
        elif isinstance(hp, CS.UniformFloatHyperparameter) or isinstance(hp, CS.UniformIntegerHyperparameter):
            info["type"] = "pow" if hp.log else ("int" if isinstance(hp, CS.UniformIntegerHyperparameter) else "num")
            info["lb"], info["ub"] = hp.lower, hp.upper
            if hp.log:
                info["base"] = 10
        else:
            raise TypeError(f"{type(type(hp))} is not supported")

        config_info.append(info)

    return DesignSpace().parse(config_info)


class HEBOOptimizer(AbstractAskTellOptimizer):
    def __init__(self, hebo_space, obj_key: str):
        self._hebo = HEBO(space=hebo_space)
        self._obj_key = obj_key
        self._count_for_debug = 0

    def ask(self) -> tuple[dict[str, Any], None, None]:
        self._count_for_debug += 1
        if self._count_for_debug % 20 == 0:
            print(f"Sample {self._count_for_debug}-th config at {time.time()}")

        config: pd.DataFrame = self._hebo.suggest()
        eval_config = {name: next(iter(v.values())) for name, v in config.to_dict().items()}
        return eval_config, None, None

    def tell(self, eval_config: dict[str, Any], results: dict[str, float], **kwargs) -> None:
        config = pd.DataFrame({name: {0: v} for name, v in eval_config.items()})
        self._hebo.observe(config, np.array([[results[self._obj_key]]]))


def run_hebo(
    obj_func: Any,
    config_space: CS.ConfigurationSpace,
    save_dir_name: str,
    seed: int,
    n_workers: int,
    tmp_dir: str | None,
    n_evals: int = 450,  # eta=3,S=2,100 full evals
):
    n_actual_evals_in_opt = n_evals + n_workers
    wrapper = ObjectiveFuncWrapper(
        obj_func=obj_func,
        n_workers=n_workers,
        save_dir_name=save_dir_name,
        n_actual_evals_in_opt=n_actual_evals_in_opt,
        n_evals=n_evals,
        seed=seed,
        ask_and_tell=True,
        tmp_dir=tmp_dir,
    )
    hebo_space = extract_space(config_space=config_space)
    hebo_opt = HEBOOptimizer(hebo_space=hebo_space, obj_key=wrapper.obj_keys[0])
    wrapper.simulate(opt=hebo_opt)


if __name__ == "__main__":
    args = parse_args()
    save_dir_name = get_save_dir_name(opt_name="hebo", args=args)
    bench = get_bench_instance(args, use_fidel=False)
    run_hebo(
        obj_func=bench,
        config_space=bench.config_space,
        n_workers=args.n_workers,
        save_dir_name=save_dir_name,
        seed=args.seed,
        tmp_dir=args.tmp_dir,
    )
