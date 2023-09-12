from __future__ import annotations

from validation.constants import DATASET_NAMES


OPENML_IDS = {
    "hpobench": [
        167104,
        167184,
        189905,
        167161,
        167181,
        167190,
        189906,
        167168,
    ],
    "lc": [
        3945,
        7593,
        34539,
        126025,
        126026,
        126029,
        146212,
        167104,
        167149,
        167152,
        167161,
        167168,
        167181,
        167184,
        167185,
        167190,
        167200,
        167201,
        168329,
        168330,
        168331,
        168335,
        168868,
        168908,
        168910,
        189354,
        189862,
        189865,
        189866,
        189873,
        189905,
        189906,
        189908,
        189909,
    ]
}


def get_target_name(bench_name: str, dataset_id: int | None) -> str:
    if dataset_id is None:
        return dict(
            branin="the Branin function",
            hartmann3d="the 3D Hartmann function",
            hartmann6d="the 6D Hartmann function",
        )[bench_name]
    elif bench_name in ["jahs", "hpolib"]:
        bn = dict(jahs="JAHS-Bench-201", hpolib="HPOlib")[bench_name]
        dataset_name = {
            "cifar10": "CIFAR10",
            "fashion-mnist": "Fashion-MNIST",
            "colorectal-histology": "Colorectal Histology",
            "slice-localization": "Slice Localization",
            "protein-structure": "Protein Structure",
            "naval-propulsion": "Naval Propulsion",
            "parkinsons-telemonitoring": "Parkinsons Telemonitoring",
        }[DATASET_NAMES[bench_name][dataset_id]]
        return f"{dataset_name} of {bn}"
    else:
        bn = dict(hpobench="HPOBench", lc="LCBench")[bench_name]
        return f"OpenML ID {OPENML_IDS[bench_name][dataset_id]} from {bn}"


def generate_figure_code(bench_name: str, dataset_id: int | None = None) -> str:
    target = bench_name if dataset_id is None else f"{bench_name}-{DATASET_NAMES[bench_name][dataset_id]}"
    return "\n".join([
        "\\begin{figure}",
        "\\centering",
        "\\includegraphics[width=0.98\\textwidth]{figs/perf-over-time/" + f"{target}" + ".pdf}",
        "\\caption{",
        "The performance over time on " + get_target_name(bench_name=bench_name, dataset_id=dataset_id) + ".",
        "}",
        "\\label{appx:additional-results:fig:" + target + "}",
        "\\end{figure}",
    ])


if __name__ == "__main__":
    bench_names = ["branin", "hartmann3d", "hartmann6d"]
    for bn in bench_names:
        print(generate_figure_code(bench_name=bn))
        print()

    for bn in ["hpobench", "hpolib", "jahs", "lc"]:
        for i in range(len(DATASET_NAMES[bn])):
            print(generate_figure_code(bench_name=bn, dataset_id=i))
            print()
