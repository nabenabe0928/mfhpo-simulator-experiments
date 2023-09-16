from __future__ import annotations

import os

import ujson as json

from validation.constants import DATASET_NAMES, OPT_DICT


BENCH_DICT = dict(
    lc="LCBench", hpolib="HPOlib", hpobench="HPOBench", jahs="JAHS-Bench-201", synfn="synthetic functions"
)


def collect_data() -> tuple[dict[str, float], dict[str, float]]:
    # opt x dataset
    sim_times = {}
    act_times = {}
    counter = 0
    for root, dir_names, file_names in os.walk("mfhpo-simulator-info/"):
        if len(file_names) == 0:
            continue

        counter += 1
        if counter % 3000 == 0:
            print(f"Checked {counter}/45480 files")

        data = json.load(open(os.path.join(root, "results.json")))
        _, opt_name, cond, _ = root.split("/")
        subcond = cond.split("_")
        if len(subcond) == 2:
            target_name = f"{opt_name}:{subcond[0][6:]}:{subcond[1][-1]}"
        else:
            bench_name = subcond[0][6:]
            dataset_name = subcond[1][8:]
            n_workers = subcond[2][-1]
            target_name = f"{opt_name}:{bench_name}:{dataset_name}:{n_workers}"

        assert data["cumtime"][-1] < 1e9  # sanity check
        if opt_name != "hebo" or cond[-1] == "1":
            assert data["actual_cumtime"][-1] < data["cumtime"][-1], root  # sanity check

        sim_times[target_name] = sim_times.get(target_name, 0.0) + data["cumtime"][-1]
        act_times[target_name] = act_times.get(target_name, 0.0) + data["actual_cumtime"][-1]

    return sim_times, act_times


def generate_table(data_part: str, opt_name: str, bench_name: str) -> str:
    headers = ["{" + f"$P={n_workers}$" + "}" for n_workers in [1, 2, 4, 8]]
    first_half = "\n".join([
        "\\addtolength{\\tabcolsep}{-5pt}",
        "\\begin{table}[t]",
        "\\begin{center}",
        "\\caption{",
        f"Actual and simulated runtimes for {OPT_DICT[opt_name]} on {BENCH_DICT[bench_name]}.",
        "}",
        "\\label{appx:additional-results:tab:" + f"{opt_name}-on-{bench_name}" + "}",
        "\\makebox[0.98 \\textwidth][c]{",
        "\\resizebox{0.98 \\textwidth}{!}{"
        "\\begin{tabular}{c|ccc|ccc|ccc|ccc|}",
        "\\toprule",
        "\\multirow{2}{*}{ID}" + "".join(["&\\multicolumn{3}{c|}" + h for h in headers]) + "\\\\",
        "&Act.& Sim.&$\\times$~Fast" * 4 + "\\\\",
        "\\midrule",
    ])
    second_half = "\n".join([
        "\\bottomrule",
        "\\end{tabular}",
        "}",
        "}",
        "\\end{center}",
        "\\end{table}",
        "\\addtolength{\\tabcolsep}{5pt}"
    ])
    return first_half + "\n" + data_part + "\n" + second_half


def get_data_part(
    opt_name: str,
    sim_times: dict[str, float],
    act_times: dict[str, float],
    suffixes: list[str],
    id_start: int = 1,
) -> str:
    rows = []
    for i, suffix in enumerate(suffixes, start=id_start):
        row = str(i)
        for n_workers in [1, 2, 4, 8]:
            key = f"{opt_name}:{suffix}:{n_workers}"
            not_exist = key not in sim_times
            st = "--" if not_exist else "e+".join(f'{sim_times[key]/30.0:.1e}'.split("e+0"))
            at = "--" if not_exist else "e+".join(f'{act_times[key]/30.0:.1e}'.split("e+0"))
            ratio = "--" if not_exist else "e+".join(f'{sim_times[key]/act_times[key]:.1e}'.split("e+0"))
            if not not_exist and sim_times[key] / act_times[key] < 1.0:
                ratio = "\\textbf{" + ratio + "}"

            row += f"&{at}&{st}&{ratio}" if not_exist else f"&{at}/&{st}/&{ratio}"
        row += "\\\\"
        rows.append(row)

    return "\n".join(rows)


def compute_overall_reduction(sim_times: dict[str, float], act_times: dict[str, float]) -> None:
    sim_total, act_total = sum(v for v in sim_times.values()), sum(v for v in act_times.values())
    print(f"{sim_total=:.3e}, {act_total=:.3e}, {(sim_total/act_total)=:.3e}")


def data_part_for_main(sim_times: dict[str, float], act_times: dict[str, float]) -> str:
    sim_total = {i: 0.0 for i in [1, 2, 4, 8]}
    act_total = {i: 0.0 for i in [1, 2, 4, 8]}
    for k in sim_times:
        n_workers = int(k[-1])
        sim_total[n_workers] += sim_times[k]
        act_total[n_workers] += act_times[k]

    row = ""
    for n_workers in [1, 2, 4, 8]:
        ratio = sim_total[n_workers] / act_total[n_workers]
        row += f"&{act_total[n_workers]:.1e}/&{sim_total[n_workers]:.1e}/&{ratio:.1e}"
    row += "\\\\"
    return row


if __name__ == "__main__":
    sim_times, act_times = collect_data()
    kwargs = dict(sim_times=sim_times, act_times=act_times)
    suffixes = ["branin", "hartmann3d", "hartmann6d"]
    for opt_name in OPT_DICT:
        data_part = get_data_part(suffixes=suffixes, opt_name=opt_name, **kwargs)
        print(generate_table(data_part=data_part, opt_name=opt_name, bench_name="synfn"))
        print()

    for bench_name in ["hpobench", "hpolib", "jahs", "lc"]:
        suffixes = [
            f"{bench_name}:{dn}" for dn in DATASET_NAMES[bench_name]
        ]
        for opt_name in OPT_DICT:
            if opt_name == "smac" and bench_name in ["jahs", "lc"]:
                continue

            data_part = get_data_part(suffixes=suffixes, opt_name=opt_name, **kwargs)
            print(generate_table(data_part=data_part, opt_name=opt_name, bench_name=bench_name))
            print()

    data_part = data_part_for_main(sim_times=sim_times, act_times=act_times)
    print(generate_table(data_part=data_part, opt_name=opt_name, bench_name=bench_name))
    compute_overall_reduction(sim_times=sim_times, act_times=act_times)
