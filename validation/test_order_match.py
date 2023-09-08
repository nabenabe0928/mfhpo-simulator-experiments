from __future__ import annotations

import json
import shutil
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait

from benchmark_simulator import ObjectiveFuncWrapper

import numpy as np


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
        unittime: float = 0.0,
    ):
        self._rng = np.random.RandomState(seed)
        self._n_workers = n_workers
        self._n_actual_evals = n_evals + n_workers
        self._n_evals = n_evals
        runtimes = getattr(self._rng, dist)(size=self._n_actual_evals, **dist_kwargs) * multiplier + 0.1
        runtimes[-n_workers:] = 10 ** 5
        self._runtimes = runtimes.tolist()[::-1]
        self._with_wrapper = with_wrapper
        self._observations = []
        self._unittime = unittime

    def ask(self) -> dict[str, int]:
        waiting_time = (len(self._observations) + 1) * self._unittime
        if len(self._runtimes):
            time.sleep(waiting_time)
            return self._runtimes.pop()
        else:
            return 10**5

    def _pop_completed(self, futures) -> None:
        completed, _ = wait(futures.keys(), return_when=FIRST_COMPLETED)
        for future in completed:
            config_id = futures[future]
            try:
                results = future.result()
            except Exception as e:
                raise RuntimeError(f"An exception occurred: {e}")
            else:
                self._observations.append((config_id, results["loss"]))

            futures.pop(future)

    def run(self, func) -> None:
        futures = {}
        counts = 0
        start = time.time()
        n_evals = self._n_actual_evals if self._with_wrapper else self._n_evals
        with ProcessPoolExecutor(max_workers=self._n_workers) as executor:
            while len(self._observations) < n_evals:
                if counts == n_evals or len(futures) >= self._n_workers:
                    self._pop_completed(futures)

                if counts < n_evals:
                    x = self.ask()
                    futures[executor.submit(func, dict(x=x, config_id=counts))] = counts
                    time.sleep(1e-3)
                    counts += 1

        cumtimes = np.array([r[1] - start for r in self._observations])
        order = np.argsort(cumtimes)
        indices = np.array([r[0] for r in self._observations])
        return indices[order], cumtimes[order]


def experiment(unittime: float = 0.0, avg_time: float = 5.0):
    results = {}
    expensive = "expensive" if unittime > 0.0 else "cheap"
    # make expectation 1 for everything except for Pareto distribution
    for dist, dist_kwargs, multiplier in zip(
        ["random", "exponential", "pareto", "lognormal"],
        [{}, {}, {"a": 1}, {}],
        [2.0, 1.0, 1.0, 1.0 / np.exp(1.5)],
    ):
        print(f"Start {dist=}")
        multiplier *= avg_time
        results[dist] = {}
        kwargs = dict(seed=0, dist=dist, dist_kwargs=dist_kwargs, multiplier=multiplier, unittime=unittime)
        opt = FixedRandomOpt(**kwargs)
        answer, _ = opt.run(dummy_func_with_sleep)
        results[dist]["answer"] = answer.tolist()

        opt = FixedRandomOpt(**kwargs, with_wrapper=True)
        func = ObjectiveFuncWrapper(
            obj_func=dummy_func,
            save_dir_name="validation-order-wrapper",
            tmp_dir="validation-results",
            store_config=True,
            # If expensive_sampler=False for expensive ones, since |D| does not catch up with the real number,
            # sampling times will be underestimated, and hence the overall runtime will be smaller.
            expensive_sampler=bool(unittime > 0.0),
        )
        opt.run(func)
        simulated_results = np.array(func.get_results()["config_id"])[:answer.size]
        results[dist]["simulated"] = simulated_results.tolist()
        shutil.rmtree("validation-results/mfhpo-simulator-info/validation-order-wrapper/")

        opt = FixedRandomOpt(**kwargs)
        random_results, _ = opt.run(dummy_func)
        results[dist]["random"] = random_results.tolist()

    if expensive == "expensive" and unittime * 200 < avg_time:
        expensive = "bit-expensive"

    with open(f"validation-results/order-match-{expensive}-results.json", mode="w") as f:
        json.dump(results, f, indent=4)


if __name__ == "__main__":
    experiment()
    experiment(unittime=0.05)
    experiment(unittime=0.005)
