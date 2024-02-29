from __future__ import annotations

import json

import matplotlib.pyplot as plt


plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 18
plt.rcParams['mathtext.fontset'] = 'stix'  # The setting of math font

OPT_DICT = {
    "random": "Random",
    "hyperband": "HyperBand",
    "tpe": "TPE",
    "bohb": "BOHB",
    "hebo": "HEBO",
    "dehb": "DEHB",
    "neps": "NePS",
    "smac": "SMAC",
}
COLOR_DICT = {
    "bohb": "blue",
    "dehb": "darkred",
    "neps": "purple",
    "tpe": "blue",
    "random": "black",
    "hyperband": "black",
    "hebo": "red",
    "smac": "green",
}
LS_DICT = {
    "bohb": "dotted",
    "dehb": "dotted",
    "neps": "dashed",
    "tpe": None,
    "random": None,
    "hyperband": "dotted",
    "hebo": None,
    "smac": "dashed",
}

DATASET_NAMES = json.load(open("utils/dataset-names.json"))


def get_name(
    opt_name: str,
    bench_name: str,
    n_workers: int,
    seed: int,
    dataset_name: str | None = None,
) -> str:
    prefix = f"mfhpo-simulator-info/{opt_name}/bench={bench_name}"
    suffix = f"_nworkers={n_workers}/{seed}"
    if dataset_name is not None:
        return prefix + f"_dataset={dataset_name}" + suffix
    else:
        return prefix + suffix


def get_all_path_list(bench_name: str, n_workers: int) -> list[list[list[str]]]:
    if bench_name in DATASET_NAMES:
        return [
            [
                [
                    get_name(
                        opt_name=opt_name,
                        seed=seed,
                        bench_name=bench_name,
                        dataset_name=dataset_name,
                        n_workers=n_workers,
                    )
                    for seed in range(30)
                ] for opt_name in OPT_DICT if opt_name != "smac" or bench_name not in ["lc", "jahs"]
            ] for dataset_name in DATASET_NAMES[bench_name]
        ]
    else:
        return [
                [
                    [
                        get_name(
                            opt_name=opt_name,
                            seed=seed,
                            bench_name=bench_name,
                            n_workers=n_workers,
                        )
                        for seed in range(30)
                    ] for opt_name in OPT_DICT
                ]
            ]
