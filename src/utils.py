from __future__ import annotations

from argparse import ArgumentParser
from dataclasses import dataclass
from typing import Any

from benchmark_apis import HPOLib, JAHSBench201, LCBench, MFBranin, MFHartmann

from benchmark_simulator import ObjectiveFuncWrapper

import ConfigSpace as CS

import optuna


BENCH_CHOICES = dict(lc=LCBench, hpolib=HPOLib, jahs=JAHSBench201, branin=MFBranin, hartmann=MFHartmann)


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
        seed=seed,
        tmp_dir=tmp_dir,
    )
    wrapper.set_config_space(config_space=config_space)
    study = optuna.create_study(sampler=sampler)
    study.optimize(wrapper, n_trials=n_actual_evals_in_opt, n_jobs=n_workers)


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
