from __future__ import annotations

import json

import matplotlib.pyplot as plt

import numpy as np


plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 18
plt.rcParams["mathtext.fontset"] = "stix"  # The setting of math font

_, ax = plt.subplots(figsize=(10, 5))

data = json.load(open("validation-results/deterministic.json"))
color_dict = {"naive": "black", "ours": "red", "ours_ask_and_tell": "blue"}
label_dict = {"naive": "Naïve", "ours": "MCS", "ours_ask_and_tell": "SCS"}
for k, v in data.items():
    cumtimes = v["simulated_cumtime"] if k != "naive" else v["actual_cumtime"]
    m, s = np.mean(cumtimes, axis=0), np.std(cumtimes, axis=0) / np.sqrt(len(cumtimes))
    dx = np.arange(m.size) + 1
    ax.plot(dx, m, color=color_dict[k], linestyle="dotted", label=label_dict[k])
    ax.fill_between(dx, m - s, m + s, alpha=0.2, color=color_dict[k])

ax.set_xlabel("# of Evaluations")
ax.set_ylabel("Simulated Cumulative Runtime [s]")
ax.set_yscale("log")
ax.set_xlim(1, dx.size)
ax.grid(which="minor", color="gray", linestyle=":")
ax.grid(which="major", color="black")
ax.legend(loc="lower right")
plt.savefig("figs/cumtime-traj.pdf", bbox_inches="tight")
