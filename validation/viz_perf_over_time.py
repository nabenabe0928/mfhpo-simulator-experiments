from __future__ import annotations

import os

from benchmark_simulator.utils import get_performance_over_time_from_paths, get_mean_and_standard_error

import numpy as np

import matplotlib.pyplot as plt

from validation.constants import COLOR_DICT, DATASET_NAMES, LS_DICT, OPT_DICT


def plot_perf_over_time(
    bench_name: str,
    dataset_id: int | None = None,
    ylim: tuple[float, float] | None = None,
    multiplier: float = 1.0,
):
    if dataset_id is None:
        prefix = f"bench={bench_name}"
        dataset_name = None
    else:
        dataset_name = DATASET_NAMES[bench_name][dataset_id]
        prefix = f"bench={bench_name}_dataset={dataset_name}"

    log = bench_name != "hpolib" and not bench_name.startswith("hartmann")
    fig, axes = plt.subplots(
        nrows=2,
        ncols=2,
        figsize=(15, 7),
        sharex=True,
        sharey=True,
        gridspec_kw=dict(wspace=0.04, hspace=0.15)
    )

    for i, n_workers in enumerate([1, 2, 4, 8]):
        lines, labels = [], []
        ax = axes[i // 2][i % 2]
        ax.set_title(f"$P = {n_workers}$")
        for opt, label in OPT_DICT.items():
            if opt == "smac" and any(bench_name.startswith(kw) for kw in ["jahs", "lc"]):
                continue

            dt, perfs = get_performance_over_time_from_paths(
                paths=[f"mfhpo-simulator-info/{opt}/{prefix}_nworkers={n_workers}/{seed}" for seed in range(30)],
                step=100,
            )
            if opt == "random" and n_workers == 1:
                xlim = (0.05 * np.min(dt), np.max(dt))

            m, s = get_mean_and_standard_error(perfs)
            m *= multiplier
            s *= multiplier
            line, = ax.plot(dt, m, color=COLOR_DICT[opt], ls=LS_DICT[opt])
            ax.fill_between(dt, m - s, m + s, alpha=0.2, color=COLOR_DICT[opt])
            ax.grid(which='minor', color='gray', linestyle=':')
            ax.grid(which='major', color='black')
            ax.set_xscale("log")
            if log:
                ax.set_yscale("log")

            ax.set_xlim(*xlim)
            if ylim is not None:
                ax.set_ylim(*ylim)

            lines.append(line)
            labels.append(label)

    fig.supxlabel("Simulated Runtime [s]", y=0)
    # fig.supylabel("Cumulative Minimum Value", x=0.06)
    axes[-1][0].legend(
        handles=lines,
        loc='upper center',
        labels=labels,
        fontsize=22,
        bbox_to_anchor=(1.0, -0.27),  # ここは調整が必要です
        fancybox=False,
        ncol=(len(labels) + 1) // 2,
    )
    bench_name += f"-{dataset_name}" if dataset_name is not None else ""
    file_name = f"figs/perf-over-time/{bench_name}.pdf"
    plt.savefig(file_name, bbox_inches="tight")


if __name__ == "__main__":
    os.makedirs("figs/perf-over-time/", exist_ok=True)
    plot_perf_over_time(bench_name="branin", ylim=(0.01, 100))
    plot_perf_over_time(bench_name="hartmann3d")
    plot_perf_over_time(bench_name="hartmann6d")

    for bench_name, dataset_names in DATASET_NAMES.items():
        for dataset_id in range(len(dataset_names)):
            print(bench_name, dataset_names[dataset_id])
            multiplier = 100.0 if bench_name in ["lc", "hpobench"] else 1.0
            plot_perf_over_time(bench_name=bench_name, dataset_id=dataset_id, multiplier=multiplier)
