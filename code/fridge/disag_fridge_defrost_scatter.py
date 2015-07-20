import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

from sklearn.covariance import EllipticEnvelope
sys.path.append("../common")

from common_functions import latexify, format_axes
import matplotlib.patches as mpatches
script_path = os.path.dirname(os.path.realpath(__file__))

DATA_PATH = os.path.join(script_path, "..","..","data/fridge")

submetered_homes_feedback = np.array([142, 144, 146, 151, 152, 155, 157, 159, 163, 167, 169, 170])

def return_name(folder):
    algo_name = folder.split("_")[-1]
    algo_N = folder.split("_")[0][1]
    algo_K = folder.split("_")[1][1]
    if algo_name=="Hart":
        return "Hart", 2, 3
    else:
        return algo_name, algo_N, algo_K

latexify(columns=2, fig_height=4.8)
algo_total_with_train=["N3_K3_T50_CO", "N3_K4_T50_CO", "N3_K5_T50_CO","N3_K6_T50_CO",
               "N4_K3_T50_CO", "N4_K4_T50_CO", "N4_K5_T50_CO","N4_K6_T50_CO",


               "N3_K3_T50_FHMM", "N3_K4_T50_FHMM", "N3_K5_T50_FHMM","N3_K6_T50_FHMM",
               "N4_K3_T50_FHMM", "N4_K4_T50_FHMM", "N4_K5_T50_FHMM","N4_K6_T50_FHMM"
               ]
algo_total = map(lambda x: x[:5]+x[9:], algo_total_with_train)
#algo_total = ["N2_K3_CO","N2_K4_CO","N2_K3_FHMM","N2_K4_FHMM", "N2_K3_Hart"]
ncols = len(algo_total)
fig, ax = plt.subplots(ncols=3,nrows=3,sharex=True, sharey=True)
output = {}
gt_df = pd.read_csv(os.path.expanduser("~/git/nilm-actionable/data/fridge/usage_defrost_cycles.csv"))

latexify(columns=2, fig_height=5.4)
fig, ax = plt.subplots(nrows=4, ncols=4, sharey=True, sharex=True)
for i, folder in enumerate(algo_total):

    algo, N, K = return_name(folder)

    df = pd.read_csv(os.path.join(DATA_PATH, "%s_usage_defrost_cycles.csv" %folder)).dropna()
    df = df[df.home.isin(gt_df.home)]
    gt_copy = gt_df.copy()
    gt_copy = gt_copy[gt_copy.home.isin(df.home)]
    x = gt_copy["defrost_percentage"]
    y = df["defrost_percentage"]
    x_sort = x.copy()
    x_sort.sort()
    y_upper = x_sort*1.1
    y_lower = x_sort*0.9
    ax[i/4][i%4].scatter(gt_copy["defrost_percentage"], df["defrost_percentage"], color="gray", alpha=0.6)

    ax[i/4][i%4].set_aspect('equal')
    ax[i/4][i%4].plot(x_sort, y_upper)
    ax[i/4][i%4].plot(x_sort, y_lower)
    percentage = ((x<y_upper) & (x>y_lower)).sum()*100.0/len(x)
    ax[i/4][i%4].set_title(algo+" N="+str(N)+" K="+str(K)+"\nPercentage of homes within\n 10 percent of submetered=%0.2f" %percentage)
    format_axes(ax[i/4][i%4])

fig.text(0.5, -0.01, 'Submetered defrost percentage', ha='center')
fig.text(-0.01, 0.5, 'Predicted defrost percentage', va='center', rotation='vertical')

fig.tight_layout()

plt.savefig(os.path.expanduser("~/git/nilm-actionable/figures/fridge/disag_defrost.pdf"),
            bbox_inches="tight")











