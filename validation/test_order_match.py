from __future__ import annotations

import json
import multiprocessing
import shutil
import time
from contextlib import contextmanager

from benchmark_simulator import ObjectiveFuncWrapper

import numpy as np


@contextmanager
def get_pool(n_workers: int) -> multiprocessing.Pool:
    pool = multiprocessing.Pool(processes=n_workers)
    yield pool
    pool.close()
    pool.join()


def dummy_func(eval_config: dict[str, int], *args, **kwargs) -> dict[str, float]:
    return dict(loss=time.time(), runtime=eval_config["x"])


def dummy_func_with_sleep(eval_config: dict[str, int], *args, **kwargs) -> dict[str, float]:
    results = dummy_func(eval_config, *args, **kwargs)
    config_id, runtime = eval_config["config_id"], results["runtime"]
    print(f"Sleep {runtime:.2f} seconds for {config_id=}")
    time.sleep(results["runtime"])
    results["loss"] = time.time()
    return results


class FixedRandomOpt:
    def __init__(
        self,
        seed: int,
        dist: str,
        dist_kwargs: dict,
        multiplier: float = 1.0,
        n_evals: int = 100,
        n_workers: int = 4,
        with_wrapper: bool = False,
    ):
        self._rng = np.random.RandomState(seed)
        self._n_workers = n_workers
        self._n_actual_evals = n_evals + n_workers
        self._n_evals = n_evals
        self._runtimes = getattr(self._rng, dist)(size=self._n_actual_evals, **dist_kwargs) * multiplier + 0.1
        self._runtimes[-n_workers:] = 10 ** 5
        self._with_wrapper = with_wrapper

    def run(self, func) -> None:
        results = []
        start = time.time()
        n_evals = self._n_actual_evals if self._with_wrapper else self._n_evals
        with get_pool(n_workers=self._n_workers) as pool:
            for idx in range(n_evals):
                r = pool.apply_async(func, kwds=dict(eval_config=dict(x=self._runtimes[idx], config_id=idx)))
                results.append(r)
            cumtimes = np.array([r.get()["loss"] - start for r in results])

        order = np.argsort(cumtimes)
        return np.arange(n_evals)[order]


if __name__ == "__main__":
    results = {}
    avg_time = 10.0
    # make expectation 1 for everything except for Pareto distribution
    for dist, dist_kwargs, multiplier in zip(
        ["random", "exponential", "pareto", "lognormal"],
        [{}, {}, {"a": 1}, {}],
        [2.0, 1.0, 1.0, 1.0 / np.exp(1.5)],
    ):
        print(f"Start {dist=}")
        multiplier *= avg_time
        results[dist] = {}
        kwargs = dict(seed=0, dist=dist, dist_kwargs=dist_kwargs, multiplier=multiplier)
        opt = FixedRandomOpt(**kwargs)
        answer = opt.run(dummy_func_with_sleep)
        results[dist]["answer"] = answer.tolist()

        opt = FixedRandomOpt(**kwargs, with_wrapper=True)
        func = ObjectiveFuncWrapper(
            obj_func=dummy_func,
            save_dir_name="validation-order-wrapper",
            tmp_dir="validation-results",
            store_config=True,
        )
        opt.run(func)
        simulated_results = np.array(func.get_results()["config_id"])[:answer.size]
        results[dist]["simulated"] = simulated_results.tolist()
        shutil.rmtree("validation-results/mfhpo-simulator-info/validation-order-wrapper/")

        opt = FixedRandomOpt(**kwargs)
        random_results = opt.run(dummy_func)
        results[dist]["random"] = random_results.tolist()

    with open("validation-results/order-match-results.json", mode="w") as f:
        json.dump(results, f, indent=4)
