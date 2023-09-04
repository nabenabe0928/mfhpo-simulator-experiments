from __future__ import annotations

import os
from argparse import ArgumentParser
from typing import Final


MAX_BIT: Final[int] = 30


def popcount(num: int, max_bit: int = MAX_BIT) -> int:
    counter = 0
    for i in range(max_bit):
        if (num >> i) & 1:
            counter += 1
    return counter


def check_files(opt_name: str) -> dict[str, int]:
    os.chdir(f"mfhpo-simulator-info/{opt_name}")
    counter: dict[str, int] = {}
    for tup in os.walk("."):
        if len(tup[-1]) and "complete.lock" in tup[-1]:
            s = tup[0].split("/")
            seed = int(s[-1])
            key = s[1]
            prev = counter.get(key, 0)
            counter[key] = prev | (1 << seed)
        elif len(tup[0]) >= 3 and "/" in tup[0][-3:]:
            key = tup[0].split("/")[1]
            counter.setdefault(key, 0)
    return counter


def list_missing_files(counter: dict[str, int], max_bit: int = MAX_BIT) -> None:
    complete_num = (1 << MAX_BIT) - 1
    for dir_name, num in counter.items():
        if num == complete_num:
            continue
        print(dir_name)
        missing_seeds = [str(i) for i in range(max_bit) if not ((num >> i) & 1)]
        print(" ".join(missing_seeds))
        print()


if __name__ == "__main__":
    parser = ArgumentParser()
    opt_choices = ["random", "hyperband", "dehb", "bohb", "hebo", "smac", "tpe", "neps"]
    parser.add_argument("--opt_name", type=str, required=True, choices=opt_choices)
    args = parser.parse_args()
    list_missing_files(check_files(args.opt_name))
