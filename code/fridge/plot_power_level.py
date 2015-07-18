import pandas as pd
import os
import glob
import numpy as np

import matplotlib.pyplot as plt

import sys

sys.path.append("../common")

from common_functions import latexify, format_axes
script_path = os.path.dirname(os.path.realpath(__file__))

DATA_PATH = os.path.join(script_path, "..", "..", "data/fridge/power_level_disag.csv")
RESULT_PATH = os.path.join(script_path, "..","..","data/fridge")

FOLDER_NAMES = ["N2_K4_T50_Hart", "N2_K4_T50_CO", "N2_K4_T50_FHMM", "N3_K4_T50_CO" , "N3_K4_T50_FHMM"]

df = pd.read_csv(DATA_PATH, index_col=0)

FOLDER_NAMES = ["N2_K4_T50_Hart", "N2_K4_T50_CO", "N2_K4_T50_FHMM", "N3_K4_T50_CO" , "N3_K4_T50_FHMM"]
latexify(columns=2, fig_height=2.6)
fig, ax = plt.subplots(ncols=5, sharey=True)

for i, folder in enumerate(FOLDER_NAMES):
    folder_path = os.path.join(DATA_PATH, folder)
    algo_name = folder.split("_")[-1]
    dictionary_key = folder[:3]+algo_name
    N = folder[1]
    gt = df["GT"].values
    upper_bound = 1.1*gt
    lower_bound = 0.9*gt
    under_pred = (df[dictionary_key]<lower_bound).sum()
    extra_pred = (df[dictionary_key]>upper_bound).sum()
    ax[i].scatter(df["GT"], df[dictionary_key],c='gray',alpha=0.5)
    ax[i].set_xlabel("GT power")
    ax[i].plot(gt, upper_bound, color='black', alpha=0.6, linewidth=2)
    ax[i].plot(gt, lower_bound, color='black', alpha=0.6, linewidth=2)

    ax[i].set_title(r"%s N=%s" "\n Missed= %d, Extra= %d" %(algo_name,N, under_pred, extra_pred))
    format_axes(ax[i])
ax[0].set_ylabel("Predicted power")
plt.tight_layout()

plt.savefig(os.path.join(script_path, "..", "..","figures/fridge/power_level_nilm.pdf"), bbox_inches="tight")
plt.savefig(os.path.join(script_path, "..","..","figures/fridge/power_level_nilm.png"), bbox_inches="tight")
