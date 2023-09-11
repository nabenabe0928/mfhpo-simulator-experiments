import json

import matplotlib.pyplot as plt


plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 18
plt.rcParams['mathtext.fontset'] = 'stix'  # The setting of math font

OPT_DICT = {
    "random": "Random",
    "hyperband": "Hyperband",
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
