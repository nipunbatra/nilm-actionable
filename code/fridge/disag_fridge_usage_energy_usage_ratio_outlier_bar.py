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

submetered_homes_feedback = np.array([ 18,  46,  51,  54,  59,  68,  76,  87, 106, 112, 116, 123, 170])

def return_name(folder):
    algo_name = folder.split("_")[-1]
    algo_N = folder.split("_")[0][1]
    algo_K = folder.split("_")[1][1]
    if algo_name=="Hart":
        return "Hart", 2, 3
    else:
        return algo_name, algo_N, algo_K

latexify(columns=2, fig_height=4.8)
algo_total_with_train=["N2_K3_T50_CO", "N2_K4_T50_CO", "N2_K5_T50_CO","N2_K6_T50_CO",
               "N3_K3_T50_CO", "N3_K4_T50_CO", "N3_K5_T50_CO","N3_K6_T50_CO",
               "N4_K3_T50_CO", "N4_K4_T50_CO", "N4_K5_T50_CO","N4_K6_T50_CO",

               "N2_K3_T50_FHMM", "N2_K4_T50_FHMM", "N2_K5_T50_FHMM","N2_K6_T50_FHMM",
               "N3_K3_T50_FHMM", "N3_K4_T50_FHMM", "N3_K5_T50_FHMM","N3_K6_T50_FHMM",
               "N4_K3_T50_FHMM", "N4_K4_T50_FHMM", "N4_K5_T50_FHMM","N4_K6_T50_FHMM",

                "N2_K3_T50_Hart"
               ]
algo_total = map(lambda x: x[:5]+x[9:], algo_total_with_train)
#algo_total = ["N2_K3_CO","N2_K4_CO","N2_K3_FHMM","N2_K4_FHMM", "N2_K3_Hart"]
ncols = len(algo_total)
fig, ax = plt.subplots(ncols=3,nrows=3,sharex=True, sharey=True)
output = {}

for i, folder in enumerate(algo_total):

    algo, N, K = return_name(folder)

    df = pd.read_csv(os.path.join(DATA_PATH, "%s_usage_defrost_cycles.csv" %folder)).dropna()

    df["usage proportion"] = df["usage_cycles"]/(df["usage_cycles"] + df["non_usage_cycles"])

    X = df["usage proportion"].values
    Y = df["usage_percentage"].values

    XY = df[["usage proportion","usage_percentage" ]].values




    # Example settings
    n_samples = len(df)
    outliers_fraction = 0.2
    clusters_separation = [0]

    # define two outlier detection tools to be compared
    classifiers = {
        "robust covariance estimator": EllipticEnvelope(contamination=.1)}

    # Compare given classifiers under given settings
    #xx, yy = np.meshgrid(np.linspace(-0.1, 1.1, 1000), np.linspace(0, 100, 1000))
    n_inliers = int((1. - outliers_fraction) * n_samples)
    n_outliers = int(outliers_fraction * n_samples)

    # Fit the problem with varying cluster separation
    np.random.seed(42)
    # Data generation


    # Fit the model with the One-Class SVM
    #plt.figure(figsize=(10, 5))

    clf = EllipticEnvelope(contamination=.1)
    # fit the data and tag outliers
    clf.fit(XY)
    y_pred = clf.decision_function(XY).ravel()
    threshold = stats.scoreatpercentile(y_pred,
                                        100 * outliers_fraction)
    y_pred = y_pred > threshold
    # plot the levels lines and the points
    #Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
    #Z = Z.reshape(xx.shape)


    df_outlier = df[~y_pred]
    df_feedback = df_outlier[(df_outlier["usage proportion"]>df["usage proportion"].median())
                & (df_outlier["usage_percentage"]>df["usage_percentage"].median())]

    feedback_homes = df_feedback["home"].values

    extra_pred = np.setdiff1d(feedback_homes, submetered_homes_feedback)
    missed = np.setdiff1d(submetered_homes_feedback, feedback_homes)



    for metric in ["Extra homes predicted", "Missed homes"]:
        if metric not in output:
            output[metric] = {}

    for metric in ["Extra homes predicted", "Missed homes"] :
        if N not in output[metric]:
            output[metric][N] = {}

    for metric in ["Extra homes predicted", "Missed homes"]:
        if K not in output[metric][N]:
            output[metric][N][K] = {}

    o_dict= {"Extra homes predicted": len(extra_pred) ,"Missed homes":len(missed)}

    if algo=="Hart":
        hart_dict = o_dict
        print algo, o_dict
        continue
    for metric in ["Extra homes predicted", "Missed homes"] :
        output[metric][N][K][algo] = o_dict[metric]
    print algo,N,K, o_dict


latexify(columns=2, fig_height=3.6)
fig, ax = plt.subplots(nrows=2, ncols=3, sharey=True, sharex=True)

for row_num, metric in enumerate(["Missed homes","Extra homes predicted"]):
    for col_num, num_states in enumerate(['2','3','4']):
        df = pd.DataFrame(output[metric][num_states]).T
        df.plot(kind="bar", ax=ax[row_num, col_num], rot=0, legend=False)
        format_axes(ax[row_num, col_num])
        ax[row_num, col_num].axhline(y=hart_dict[metric], linestyle="--",color='red', label="Hart")
        #ax[row_num, col_num].axhline(y=gt[metric], linestyle="-",color='black', label="Submetered")
        #ax[row_num, col_num].axhline(y=hart[metric], linestyle="--",color='red', label="Hart")

        ax[0, col_num].set_title("N=%s" %num_states)
        ax[-1, col_num].set_xlabel("Top-K")


ax[0,0].set_ylabel("Missed homes\n(out of 12)")

ax[1,0].set_ylabel("Extra homes predicted\n(out of 84)")
#ax[0,0].set_ylim((0,8))
#ax[1,0].set_ylim((0,5))

co_patch = mpatches.Patch(color='blue', label='CO')
fhmm_patch =  mpatches.Patch(color='green', label='FHMM')
hart_patch = mpatches.Patch(color='red', label='Hart',lw='0.6')
fig.tight_layout()

fig.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05),
          ncol=4,handles=[co_patch, fhmm_patch, hart_patch],
           labels=["CO","FHMM","Hart"])

plt.savefig("../../figures/fridge/disag_usage_energy_ratio_all.png", bbox_inches="tight")
plt.savefig("../../figures/fridge/disag_usage_energy_ratio_all.pdf", bbox_inches="tight")

