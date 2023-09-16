from __future__ import annotations

from benchmark_simulator.utils import get_average_rank

import os

import matplotlib.pyplot as plt

from validation.constants import COLOR_DICT, LS_DICT, OPT_DICT, get_all_path_list


def plot_average_rank(bench_name: str):
    fig, axes = plt.subplots(
        nrows=2,
        ncols=2,
        figsize=(15, 7),
        sharex=True,
        sharey=True,
        gridspec_kw=dict(wspace=0.04, hspace=0.15)
    )
    for i, n_workers in enumerate([1, 2, 4, 8]):
        ax = axes[i // 2][i % 2]
        ax.set_title(f"$P = {n_workers}$")
        avg_rank, dt = get_average_rank(all_path_list=get_all_path_list(bench_name=bench_name, n_workers=n_workers))
        lines, labels = [], []
        for opt_name, r in zip(OPT_DICT, avg_rank):
            if opt_name == "smac" and bench_name in ["lc", "jahs"]:
                continue

            line, = ax.plot(dt, r, color=COLOR_DICT[opt_name], ls=LS_DICT[opt_name])
            lines.append(line)
            labels.append(OPT_DICT[opt_name])

        ax.set_xscale("log")
        ax.grid(which='minor', color='gray', linestyle=':')
        ax.grid(which='major', color='black')

    axes[-1][0].legend(
        handles=lines,
        loc='upper center',
        labels=labels,
        fontsize=22,
        bbox_to_anchor=(1.0, -0.28),  # ここは調整が必要です
        fancybox=False,
        ncol=(len(labels) + 1) // 2,
    )
    fig.supxlabel("Used Budget Ratio (Used Budget / Max. Budget)", y=0.0)
    fig.supylabel("Average Rank", x=0.08)
    plt.savefig(f"figs/avg-rank/{bench_name}.pdf", bbox_inches="tight")


if __name__ == "__main__":
    os.makedirs("figs/avg-rank", exist_ok=True)
    for bench_name in ["hpobench", "hpolib", "jahs", "lc"]:
        print(bench_name)
        plot_average_rank(bench_name=bench_name)
