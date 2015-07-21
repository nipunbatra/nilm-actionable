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

#FOLDER_NAMES = ["N2_K4_T50_Hart", "N2_K4_T50_CO", "N2_K4_T50_FHMM", "N3_K4_T50_CO" , "N3_K4_T50_FHMM"]

df = pd.read_csv(DATA_PATH, index_col=0)

#FOLDER_NAMES = ["N2_K4_T50_Hart", "N2_K4_T50_CO", "N2_K4_T50_FHMM", "N3_K4_T50_CO" , "N3_K4_T50_FHMM"]
FOLDER_NAMES = ["N3_K4_T50_CO" , "N3_K4_T50_FHMM","N2_K4_T50_Hart"]
latexify(columns=2, fig_height=2.6)
fig, ax = plt.subplots(ncols=len(FOLDER_NAMES), sharey=True)

for i, folder in enumerate(FOLDER_NAMES):
    folder_path = os.path.join(DATA_PATH, folder)
    algo_name = folder.split("_")[-1]
    dictionary_key = folder[:3]+algo_name
    N = folder[1]
    gt = df["GT"].values
    upper_bound = 1.1*gt
    lower_bound = 0.9*gt
    under_pred_df = df[df[dictionary_key]<lower_bound]
    extra_pred_df = df[df[dictionary_key]>upper_bound]
    range_df = df[(df[dictionary_key]>=lower_bound) & (df[dictionary_key]<=upper_bound)]
    under_pred = (df[dictionary_key]<lower_bound).sum()
    extra_pred = (df[dictionary_key]>upper_bound).sum()
    ax[i].scatter(range_df["GT"], range_df[dictionary_key],c='gray',alpha=0.5,edgecolors=None,lw=0.1)
    ax[i].scatter(under_pred_df["GT"], under_pred_df[dictionary_key],c='green',alpha=0.5, zorder=10,lw=0.1)
    ax[i].scatter(extra_pred_df["GT"], extra_pred_df[dictionary_key],c='green',alpha=0.5, zorder=10,lw=0.1)
    ax[i].set_xlabel("GT power")
    ax[i].plot(gt, upper_bound, color='black', alpha=0.6, linewidth=0.4)
    ax[i].plot(gt, lower_bound, color='black', alpha=0.6, linewidth=0.4)

    ax[i].set_title(r"%s" "\n FP= %d" %(algo_name,under_pred +extra_pred))
    format_axes(ax[i])
    ax[i].set_aspect('equal')
ax[0].set_ylabel("Predicted power")
plt.tight_layout()

plt.savefig(os.path.join(script_path, "..", "..","figures/fridge/power_level_nilm.pdf"), bbox_inches="tight")
plt.savefig(os.path.join(script_path, "..","..","figures/fridge/power_level_nilm.png"), bbox_inches="tight")
