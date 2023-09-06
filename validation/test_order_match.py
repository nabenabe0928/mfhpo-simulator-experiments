from __future__ import annotations

import multiprocessing
import time
from contextlib import contextmanager

import numpy as np


@contextmanager
def get_pool(n_workers: int) -> multiprocessing.Pool:
    pool = multiprocessing.Pool(processes=n_workers)
    yield pool
    pool.close()
    pool.join()


def duummy_func():
    # TODO
    pass


def dummy_func_with_sleep():
    # TODO
    results = duummy_func()
    time.sleep(results["runtime"])
    pass


class FixedRandomOpt:
    def __init__(self, seed: int, n_evals: int = 100, n_workers: int = 4):
        self._rng = np.random.RandomState(seed)
        self._n_workers = n_workers
        self._n_actual_evals = n_evals + n_workers
        self._n_evals = n_evals
        self._runtimes = self._rng.random(size=n_evals) * 10.0

    def run(self, func) -> None:
        results = []
        with get_pool(n_workers=self._n_workers) as pool:
            for config_id in range(self._n_actual_evals):
                r = pool.apply_async(func, kwds=dict(eval_config=dict(config_id=config_id)))
                results.append(r)
            for r in results:
                r.get()


if __name__ == "__main__":
    # TODO: Run several experiments with and without wrapper.
    pass
