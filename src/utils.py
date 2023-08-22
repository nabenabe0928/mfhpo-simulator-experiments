from __future__ import annotations

import os
import shutil
import sys
from argparse import ArgumentParser
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from benchmark_apis import HPOBench, HPOLib, JAHSBench201, LCBench, MFBranin, MFHartmann

from benchmark_simulator import ObjectiveFuncWrapper, get_multiple_wrappers

import ConfigSpace as CS

try:
    from hpbandster.core import nameserver as hpns
    from hpbandster.core.worker import Worker
    from hpbandster.optimizers import BOHB, HyperBand
except ModuleNotFoundError:
    class Worker:
        NotImplemented

try:
    import optuna
except ModuleNotFoundError:
    pass

try:
    from smac import HyperbandFacade as HBFacade
    from smac import MultiFidelityFacade as MFFacade
    from smac import Scenario
    from smac.main.config_selector import ConfigSelector
    from smac.intensifier.hyperband import Hyperband
except ModuleNotFoundError:
    pass

import ujson as json


BENCH_CHOICES = dict(
    lc=LCBench, hpobench=HPOBench, hpolib=HPOLib, jahs=JAHSBench201, branin=MFBranin, hartmann=MFHartmann
)
N_EVALS_DICT = dict(
    hyperband=4500,
    random=2000,
    hebo=200,
    tpe=200,
    bohb=450,
    dehb=450,
    smac=450,
    neps=450,
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
    n_evals: int = 200,
) -> None:
    n_actual_evals_in_opt = n_evals + n_workers
    wrapper = OptunaObjectiveFuncWrapper(
        obj_func=obj_func,
        n_workers=n_workers,
        save_dir_name=save_dir_name,
        n_actual_evals_in_opt=n_actual_evals_in_opt,
        n_evals=n_evals,
        max_waiting_time=120.0,
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
    load_every_call: bool,
    tmp_dir: str | None,
    n_init_min: int = 5,
    n_evals: int = 450,  # eta=3,S=2,100 full evals
) -> None:
    data_to_scatter = None
    # if not load_every_call and hasattr(obj_func, "get_benchdata"):
    #     # This data is shared in memory, and thus the optimization becomes quicker!
    #     data_to_scatter = {"benchdata": obj_func.get_benchdata()}

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
        max_waiting_time=120.0,
        fidel_keys=[fidel_key],
        continual_max_fidel=max_fidel,
        tmp_dir=tmp_dir,
    )

    Facade = HBFacade if sampler == "hyperband" else MFFacade

    class _WrappedFacade(Facade):
        @staticmethod
        def get_config_selector(
            scenario: Scenario,
            *,
            retrain_after: int = 8,
            retries: int = 1000,  # To prevent the early stopping in SMAC
        ) -> ConfigSelector:
            return ConfigSelector(scenario, retrain_after=retrain_after, retries=retries)

    smac = _WrappedFacade(
        scenario,
        wrapper.__call__,  # SMAC raises an error when using wrapper, so we use wrapper.__call__ instead.
        initial_design=MFFacade.get_initial_design(scenario, n_configs=max(n_init_min, n_workers)),
        intensifier=Hyperband(scenario, incumbent_selection="highest_budget"),
        overwrite=True,
    )

    # data_to_scatter must be a keyword argument.
    smac.optimize(data_to_scatter=data_to_scatter)


class BOHBWorker(Worker):
    # https://github.com/automl/HpBandSter
    def __init__(self, worker: ObjectiveFuncWrapper, sleep_interval: int = 0.5, **kwargs: Any):
        super().__init__(**kwargs)
        self.sleep_interval = sleep_interval
        self._worker = worker

    def compute(self, config: dict[str, Any], budget: int, **kwargs: Any) -> dict[str, float]:
        fidel_keys = self._worker.fidel_keys
        fidels = dict(epoch=int(budget)) if "epoch" in fidel_keys else {k: int(budget) for k in fidel_keys}
        # config_id: a triplet of ints(iteration, budget index, running index) internally used in BOHB
        # By passing config_id, it increases the safety in the continual learning
        config_id = kwargs["config_id"][0] + 100000 * kwargs["config_id"][2]
        results = self._worker(eval_config=config, fidels=fidels, config_id=config_id)
        return dict(loss=results["loss"])


def get_bohb_workers(
    run_id: str,
    ns_host: str,
    obj_func: Any,
    save_dir_name: str,
    max_fidel: int,
    fidel_key: str,
    n_workers: int,
    n_actual_evals_in_opt: int,
    n_evals: int,
    seed: int,
    tmp_dir: str | None,
) -> list[BOHBWorker]:
    kwargs = dict(
        obj_func=obj_func,
        n_workers=n_workers,
        save_dir_name=save_dir_name,
        continual_max_fidel=max_fidel,
        fidel_keys=[fidel_key],
        n_actual_evals_in_opt=n_actual_evals_in_opt,
        n_evals=n_evals,
        seed=seed,
        tmp_dir=tmp_dir,
    )
    bohb_workers = []
    for i, w in enumerate(get_multiple_wrappers(**kwargs, max_waiting_time=120.0)):
        worker = BOHBWorker(worker=w, id=i, nameserver=ns_host, run_id=run_id)
        worker.run(background=True)
        bohb_workers.append(worker)

    return bohb_workers


def run_bohb(
    obj_func: Any,
    config_space: CS.ConfigurationSpace,
    save_dir_name: str,
    min_fidel: int,
    max_fidel: int,
    fidel_key: str,
    seed: int,
    n_workers: int,
    tmp_dir: str | None,
    sampler: Literal["hyperband", "bohb"],
    run_id: str = "bohb-run",
    ns_host: str = "127.0.0.1",
    n_evals: int = 450,  # eta=3,S=2,100 full evals
    n_brackets: int = 72,  # 22 HB iter --> 33 SH brackets
) -> None:
    ns = hpns.NameServer(run_id=run_id, host=ns_host, port=None)
    ns.start()
    _ = get_bohb_workers(
        run_id=run_id,
        ns_host=ns_host,
        obj_func=obj_func,
        save_dir_name=save_dir_name,
        max_fidel=max_fidel,
        fidel_key=fidel_key,
        n_workers=n_workers,
        n_actual_evals_in_opt=n_evals + n_workers,
        n_evals=n_evals,
        seed=seed,
        tmp_dir=tmp_dir,
    )
    sampler_cls = HyperBand if sampler == "hyperband" else BOHB
    opt = sampler_cls(
        configspace=config_space,
        run_id=run_id,
        min_budget=min_fidel,
        max_budget=max_fidel,
    )
    opt.run(n_iterations=n_brackets, min_n_workers=n_workers)
    opt.shutdown(shutdown_workers=True)
    ns.shutdown()


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


def is_completed(save_dir_name: str, opt_name: str) -> bool:
    result_path = os.path.join("mfhpo-simulator-info", save_dir_name, "results.json")
    if not os.path.exists(result_path):
        return False

    n_evals = N_EVALS_DICT[opt_name]
    with open(result_path, mode="r") as f:
        results = json.load(f)
        return len(results) != 0 and len(results["cumtime"]) >= n_evals


def compress_files():
    prefix = "mfhpo-simulator-info/"
    lock_fn = "compress.lock"
    complete_count = {}
    for count, loc in enumerate(os.walk(prefix), start=1):
        dir_path, dir_names, file_names = loc
        if "results.json" not in file_names:
            continue

        if count % 1000 == 0:
            print(f"Checked {count} directories")

        save_dir_name = dir_path.split(prefix)[-1]
        opt_name = save_dir_name.split("/")[0]
        complete_count[opt_name] += 1
        if lock_fn in file_names:
            continue

        with open(os.path.join(prefix, save_dir_name, lock_fn), mode="w"):
            pass
        if not is_completed(save_dir_name=save_dir_name, opt_name=opt_name):
            continue

        print(f"Compress {save_dir_name}")
        for target, keys in zip(["results", "sampled_time"], [["cumtime", "loss"], ["before_sample", "after_sample"]]):
            json_fn = f"{target}.json"
            with open(os.path.join(prefix, save_dir_name, json_fn), mode="r") as f:
                data = json.load(f)
                data[keys[0]] = [float(f"{d:.6e}") for d in data[keys[0]]]
                data[keys[1]] = [float(f"{d:.6e}") for d in data[keys[1]]]
            with open(os.path.join(prefix, save_dir_name, json_fn), mode="w") as f:
                json.dump(data, f)

    print(complete_count)


def remove_failed_files():
    prefix = "mfhpo-simulator-info/"
    lock_fn = "complete.lock"
    complete_count = {}
    for count, loc in enumerate(os.walk(prefix), start=1):
        dir_path, dir_names, file_names = loc
        if "results.json" not in file_names:
            continue

        if count % 1000 == 0:
            print(f"Checked {count} directories")

        save_dir_name = dir_path.split(prefix)[-1]
        opt_name = save_dir_name.split("/")[0]
        complete_count[opt_name] += 1
        if lock_fn in file_names:
            continue

        if is_completed(save_dir_name=save_dir_name, opt_name=opt_name):
            with open(os.path.join(prefix, save_dir_name, lock_fn), mode="w"):
                pass

            continue

        print(f"Remove {save_dir_name}")
        shutil.rmtree(dir_path)

    print(complete_count)


def cleanup_info():
    prefix = "mfhpo-simulator-info/"
    count = 0
    protected = ["results.json", "compress.lock", "complete.lock", "sampled_time.json"]
    for loc in os.walk(prefix):
        dir_path, dir_names, file_names = loc
        if "results.json" not in file_names:
            continue

        count += 1
        if count % 1000 == 0:
            print(f"Checked {count} directories")

        for fn in file_names:
            if any(fn.endswith(pattern) for pattern in protected):
                continue

            os.remove(os.path.join(dir_path, fn))


def get_save_dir_name(opt_name: str, args: ParsedArgs) -> str:
    dataset_part = ""
    if BENCH_CHOICES[args.bench_name]._BENCH_TYPE == "HPO":
        dataset_name = "-".join(BENCH_CHOICES[args.bench_name]._CONSTS.dataset_names[args.dataset_id].split("_"))
        dataset_part = f"_dataset={dataset_name}"

    bench_name = args.bench_name
    if args.bench_name == "hartmann":
        bench_name = f"{args.bench_name}{args.dim}d"

    save_dir_name = f"{opt_name}/bench={bench_name}{dataset_part}_nworkers={args.n_workers}/{args.seed}"
    if is_completed(save_dir_name, opt_name=opt_name):
        sys.exit("The completed result already exists")

    return save_dir_name


def get_bench_instance(
    args: ParsedArgs, keep_benchdata: bool = True, use_fidel: bool = True, load_every_call: bool = False,
) -> Any:
    bench_cls = BENCH_CHOICES[args.bench_name]
    if bench_cls._BENCH_TYPE == "HPO":
        obj_func = bench_cls(
            dataset_id=args.dataset_id, seed=args.seed, keep_benchdata=keep_benchdata, load_every_call=load_every_call
        )
    else:
        kwargs = dict(dim=args.dim) if args.bench_name == "hartmann" else dict()
        obj_func = bench_cls(seed=args.seed, use_fidel=use_fidel, **kwargs)

    return obj_func
