from __future__ import annotations

import os

from benchmark_simulator.utils import (
    get_average_rank,
    get_performance_over_time_with_same_time_scale,
)

from validation.constants import get_all_path_list, OPT_DICT

import matplotlib.pyplot as plt

import numpy as np

import scikit_posthocs as sp


MAX_POWER_FACTOR = 10


def rank_test_with_n_workers(ax: plt.Axes, n_workers: int, budget_index: int, with_smac: bool) -> None:
    budget_prop = [1.0 / (1 << i) for i in reversed(range(MAX_POWER_FACTOR + 1))]
    bench_names = ["hpolib", "hpobench", "jahs", "lc", "branin", "hartmann3d", "hartmann6d"]
    all_path_list = []
    for bench_name in bench_names:
        if with_smac and bench_name in ["jahs", "lc"]:
            continue

        if with_smac or bench_name in ["jahs", "lc"]:
            paths = get_all_path_list(bench_name=bench_name, n_workers=n_workers)
        else:
            paths = [p[:-1] for p in get_all_path_list(bench_name=bench_name, n_workers=n_workers)]

        all_path_list.extend(paths)

    results, frac = get_performance_over_time_with_same_time_scale(all_path_list=all_path_list)
    avg_rank, _ = get_average_rank(all_path_list=all_path_list)
    indices = np.searchsorted(frac, budget_prop)
    budget, idx = budget_prop[budget_index], indices[budget_index]
    print(f"Plot for {budget=} with {n_workers=}")
    samples, ranks = results[..., idx], avg_rank[..., idx]
    test_results = sp.posthoc_conover_friedman(samples)
    sp.critical_difference_diagram(
        ranks={opt_name: r for opt_name, r in zip(OPT_DICT.values(), ranks)},
        sig_matrix=test_results,
        ax=ax,
        label_fmt_left="{label} [{rank:.2f}]  ",
        label_fmt_right="  [{rank:.2f}] {label}",
        text_h_margin=0.3,
        label_props={"color": "black", "fontweight": "bold"},
        crossbar_props={"color": "red", "marker": "o"},
        marker_props={"marker": "*", "s": 150, "color": "y", "edgecolor": "k"},
        elbow_props={"color": "gray"},
    )


def plot_critical_difference(budget_index: int, with_smac: bool = True) -> None:
    fig, axes = plt.subplots(
        nrows=2,
        ncols=2,
        sharex=True,
        sharey=True,
        figsize=(18, 7),
        gridspec_kw=dict(wspace=0.8, hspace=0.8),
    )
    for i, n_workers in enumerate([1, 2, 4, 8]):
        r, c = i // 2, i % 2
        ax = axes[r][c]
        ax.set_title(f"$P = {n_workers}$")
        rank_test_with_n_workers(ax=ax, n_workers=n_workers, budget_index=budget_index, with_smac=with_smac)

    factor = 1 << (MAX_POWER_FACTOR - budget_index)
    if with_smac:
        plt.savefig(f"figs/rank-test/with-smac/1-by-{factor}.pdf", bbox_inches="tight")
    else:
        plt.savefig(f"figs/rank-test/without-smac/1-by-{factor}.pdf", bbox_inches="tight")


if __name__ == "__main__":
    os.makedirs("figs/rank-test/with-smac", exist_ok=True)
    os.makedirs("figs/rank-test/without-smac", exist_ok=True)
    for budget_index in range(MAX_POWER_FACTOR + 1):
        plot_critical_difference(budget_index=budget_index, with_smac=True)
        plot_critical_difference(budget_index=budget_index, with_smac=False)
