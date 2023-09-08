from __future__ import annotations

import json
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from typing import Literal

from benchmark_simulator import ObjectiveFuncWrapper

import numpy as np


def dummy_func(eval_config: dict[str, int], *args, **kwargs) -> dict[str, float]:
    return dict(loss=0.0, runtime=eval_config["x"])


class Optimizer:
    def __init__(
        self,
        test_case_key: str,
        n_workers: int = 4,
        unittime: float = 10.0,
    ):
        test_case = json.load(open("validation/test-cases.json"))[test_case_key]
        self._runtimes = test_case["runtime"][::-1]
        self._answer = np.array(test_case["answer"])
        self._n_workers = n_workers
        self._n_evals = self._answer.size + self._n_workers
        self._n_observations = 0
        self._unittime = unittime

    def ask(self) -> dict[str, int]:
        waiting_time = (self._n_observations + 1) * self._unittime
        if len(self._runtimes):
            time.sleep(waiting_time)
            return self._runtimes.pop()
        else:
            return 10**5

    def _pop_completed(self, futures) -> None:
        completed, _ = wait(futures.keys(), return_when=FIRST_COMPLETED)
        for future in completed:
            try:
                future.result()
            except Exception as e:
                raise RuntimeError(f"An exception occurred: {e}")
            else:
                self._n_observations += 1

            futures.pop(future)

    def run(self, func) -> None:
        futures = {}
        counts = 0
        with ProcessPoolExecutor(max_workers=self._n_workers) as executor:
            while self._n_observations < self._n_evals:
                if counts == self._n_evals or len(futures) >= self._n_workers:
                    self._pop_completed(futures)

                if counts < self._n_evals:
                    x = self.ask()
                    futures[executor.submit(func, dict(x=x))] = None
                    time.sleep(1e-3)
                    counts += 1


class CheapOpt:
    def __init__(self):
        self._opt = Optimizer(test_case_key="cheap", unittime=0.0)

    @property
    def n_evals(self) -> int:
        return self._opt._answer.size

    @property
    def n_actual_evals(self) -> int:
        return self._opt._n_evals

    @property
    def answer(self) -> np.ndarray:
        return self._opt._answer

    def run(self, func) -> None:
        self._opt.run(func)


class ExpensiveOpt:
    def __init__(self, test_case_key: Literal["basic", "no-overlap"]):
        self._opt = Optimizer(test_case_key=f"expensive::{test_case_key}")

    @property
    def n_evals(self) -> int:
        return self._opt._answer.size

    @property
    def n_actual_evals(self) -> int:
        return self._opt._n_evals

    @property
    def answer(self) -> np.ndarray:
        return self._opt._answer

    def run(self, func) -> None:
        self._opt.run(func)


if __name__ == "__main__":
    opt = CheapOpt()
    wrapper = ObjectiveFuncWrapper(
        obj_func=dummy_func,
        expensive_sampler=False,
        n_evals=opt.n_evals,
        n_actual_evals_in_opt=opt.n_actual_evals,
        save_dir_name="validation-cheap",
        tmp_dir="validation-results",
    )
    opt.run(wrapper)

    opt = ExpensiveOpt(test_case_key="basic")
    wrapper = ObjectiveFuncWrapper(
        obj_func=dummy_func,
        expensive_sampler=True,
        n_evals=opt.n_evals,
        n_actual_evals_in_opt=opt.n_actual_evals,
        save_dir_name="validation-expensive-basic",
        tmp_dir="validation-results",
    )
    opt.run(wrapper)

    opt = ExpensiveOpt(test_case_key="no-overlap")
    wrapper = ObjectiveFuncWrapper(
        obj_func=dummy_func,
        expensive_sampler=True,
        n_evals=opt.n_evals,
        n_actual_evals_in_opt=opt.n_actual_evals,
        save_dir_name="validation-expensive-no-overlap",
        tmp_dir="validation-results",
    )
    opt.run(wrapper)
