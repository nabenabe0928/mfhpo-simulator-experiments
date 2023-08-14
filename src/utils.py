from __future__ import annotations

import os
from argparse import ArgumentParser
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from benchmark_apis import HPOBench, HPOLib, JAHSBench201, LCBench, MFBranin, MFHartmann

from benchmark_simulator import ObjectiveFuncWrapper

import ConfigSpace as CS

try:
    import optuna
except ModuleNotFoundError:
    pass

try:
    from smac import HyperbandFacade as HBFacade
    from smac import MultiFidelityFacade as MFFacade
    from smac import Scenario
    from smac.intensifier.hyperband import Hyperband
except ModuleNotFoundError:
    pass


BENCH_CHOICES = dict(
    lc=LCBench, hpobench=HPOBench, hpolib=HPOLib, jahs=JAHSBench201, branin=MFBranin, hartmann=MFHartmann
)


class OptunaObjectiveFuncWrapper(ObjectiveFuncWrapper):
    def set_config_space(self, config_space: CS.ConfigurationSpace) -> None:
        self.config_space = config_space

    def __call__(
        self,
        trial: optuna.Trial,
    ) -> float:
        eval_config: dict[str, Any] = {}
        for name in self.config_space:
            hp = self.config_space.get_hyperparameter(name)
            if isinstance(hp, CS.CategoricalHyperparameter):
                eval_config[name] = trial.suggest_categorical(name, choices=hp.choices)
            elif isinstance(hp, CS.UniformFloatHyperparameter) or hp.log:
                dtype = float if isinstance(hp, CS.UniformFloatHyperparameter) else int
                eval_config[name] = dtype(trial.suggest_float(name, low=hp.lower, high=hp.upper, log=hp.log))
            elif isinstance(hp, CS.UniformIntegerHyperparameter):
                eval_config[name] = trial.suggest_int(name, low=hp.lower, high=hp.upper)
            else:
                raise ValueError(f"{type(hp)} is not supported.")

        output = super().__call__(eval_config)
        return output[self.obj_keys[0]]


def run_optuna(
    obj_func: Any,
    config_space: CS.ConfigurationSpace,
    save_dir_name: str,
    seed: int,
    n_workers: int,
    sampler: optuna.samplers.BaseSampler,
    tmp_dir: str | None,
    n_evals: int = 450,  # eta=3,S=2,100 full evals
) -> None:
    n_actual_evals_in_opt = n_evals + n_workers
    wrapper = OptunaObjectiveFuncWrapper(
        obj_func=obj_func,
        n_workers=n_workers,
        save_dir_name=save_dir_name,
        n_actual_evals_in_opt=n_actual_evals_in_opt,
        n_evals=n_evals,
        max_waiting_time=5.0,
        seed=seed,
        tmp_dir=tmp_dir,
    )
    wrapper.set_config_space(config_space=config_space)
    study = optuna.create_study(sampler=sampler)
    study.optimize(wrapper, n_trials=n_actual_evals_in_opt, n_jobs=n_workers)


class SMACObjectiveFuncWrapper(ObjectiveFuncWrapper):
    def __call__(
        self,
        config: CS.Configuration,
        budget: int,
        seed: int | None = None,
        data_to_scatter: dict[str, Any] | None = None,
    ) -> float:
        data_to_scatter = {} if data_to_scatter is None else data_to_scatter
        eval_config = dict(config)
        output = super().__call__(eval_config, fidels={self.fidel_keys[0]: int(budget)}, **data_to_scatter)
        return output[self.obj_keys[0]]


def run_smac(
    obj_func: Any,
    config_space: CS.ConfigurationSpace,
    save_dir_name: str,
    min_fidel: int,
    max_fidel: int,
    fidel_key: list[str],
    seed: int,
    n_workers: int,
    sampler: Literal["smac", "hyperband"],
    tmp_dir: str | None,
    n_init_min: int = 5,
    n_evals: int = 450,  # eta=3,S=2,100 full evals
) -> None:
    n_actual_evals_in_opt = n_evals + n_workers
    scenario = Scenario(
        config_space,
        n_trials=n_actual_evals_in_opt,
        min_budget=min_fidel,
        max_budget=max_fidel,
        n_workers=n_workers,
        output_directory=Path(os.path.join("" if tmp_dir is None else tmp_dir, "logs/smac3")),
    )
    wrapper = SMACObjectiveFuncWrapper(
        obj_func=obj_func,
        n_workers=n_workers,
        save_dir_name=save_dir_name,
        n_actual_evals_in_opt=n_actual_evals_in_opt,
        n_evals=n_evals,
        seed=seed,
        max_waiting_time=5.0,
        fidel_keys=[fidel_key],
        continual_max_fidel=max_fidel,
        tmp_dir=tmp_dir,
    )

    facade = HBFacade if sampler == "hyperband" else MFFacade
    smac = facade(
        scenario,
        wrapper.__call__,  # SMAC raises an error when using wrapper, so we use wrapper.__call__ instead.
        initial_design=MFFacade.get_initial_design(scenario, n_configs=max(n_init_min, n_workers)),
        intensifier=Hyperband(scenario, incumbent_selection="highest_budget"),
        overwrite=True,
    )
    data_to_scatter = None
    if hasattr(obj_func, "get_benchdata"):
        # This data is shared in memory, and thus the optimization becomes quicker!
        data_to_scatter = {"benchdata": obj_func.get_benchdata()}

    # data_to_scatter must be a keyword argument.
    smac.optimize(data_to_scatter=data_to_scatter)


@dataclass(frozen=True)
class ParsedArgs:
    seed: int
    dataset_id: int
    dim: int
    bench_name: str
    n_workers: int
    worker_index: int | None
    tmp_dir: str | None


def parse_args() -> ParsedArgs:
    parser = ArgumentParser()
    parser.add_argument("--seed", type=int)
    parser.add_argument("--dataset_id", type=int, default=0, choices=list(range(34)))
    parser.add_argument("--dim", type=int, default=3, choices=[3, 6], help="Only for Hartmann")
    parser.add_argument("--bench_name", type=str, choices=list(BENCH_CHOICES.keys()))
    parser.add_argument("--n_workers", type=int)
    parser.add_argument("--tmp_dir", type=str, default=None)
    parser.add_argument("--worker_index", type=int, default=None)
    args = parser.parse_args()

    kwargs = {k: getattr(args, k) for k in ParsedArgs.__annotations__.keys()}
    return ParsedArgs(**kwargs)


def get_save_dir_name(args: ParsedArgs) -> str:
    dataset_part = ""
    if BENCH_CHOICES[args.bench_name]._BENCH_TYPE == "HPO":
        dataset_name = "-".join(BENCH_CHOICES[args.bench_name]._CONSTS.dataset_names[args.dataset_id].split("_"))
        dataset_part = f"_dataset={dataset_name}"

    bench_name = args.bench_name
    if args.bench_name == "hartmann":
        bench_name = f"{args.bench_name}{args.dim}d"

    return f"bench={bench_name}{dataset_part}_nworkers={args.n_workers}/{args.seed}"


def get_bench_instance(args: ParsedArgs, keep_benchdata: bool = True, use_fidel: bool = True) -> Any:
    bench_cls = BENCH_CHOICES[args.bench_name]
    if bench_cls._BENCH_TYPE == "HPO":
        obj_func = bench_cls(dataset_id=args.dataset_id, seed=args.seed, keep_benchdata=keep_benchdata)
    else:
        kwargs = dict(dim=args.dim) if args.bench_name == "hartmann" else dict()
        obj_func = bench_cls(seed=args.seed, use_fidel=use_fidel, **kwargs)

    return obj_func
