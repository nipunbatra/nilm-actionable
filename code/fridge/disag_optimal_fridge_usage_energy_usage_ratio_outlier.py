import sys

import pandas as pd

sys.path.append("../common")

from common_functions import latexify, format_axes

algos = ["CO", "FHMM", "Hart"]
import os
script_path = os.path.dirname(os.path.realpath(__file__))
import matplotlib.pyplot as plt
import numpy as np

submetered_homes_feedback = np.array([ 18,  46,  51,  54,  59,  68,
                                       76,  87, 106, 112, 116, 123, 170])
DATA_PATH = os.path.join(script_path, "..","..","data/fridge")

from collections import OrderedDict

path_dict = OrderedDict()
path_dict["Submetered"] = "usage_defrost_cycles.csv"
path_dict["CO"] = "N2_K5_CO_usage_defrost_cycles.csv"
path_dict["FHMM"] = "N4_K3_FHMM_usage_defrost_cycles.csv"
path_dict["Hart"]="N2_k3_Hart_usage_defrost_cycles.csv"


latexify(columns=2, fig_height=2.6)
fig, ax = plt.subplots(ncols=4, sharey=True)

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

from sklearn.covariance import EllipticEnvelope

for i, (algo_name, algo_path) in enumerate(path_dict.iteritems()):
    df = pd.read_csv(os.path.join(DATA_PATH,algo_path)).dropna()

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
    xx, yy = np.meshgrid(np.linspace(0,1, 1000), np.linspace(0,100, 1000))
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
    Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)



    subplot = ax[i]
    #subplot.contourf(xx, yy, Z, levels=np.linspace(Z.min(), threshold, 7),
    #                  cmap=plt.cm.Blues_r)
    a = subplot.contour(xx, yy, Z, levels=[threshold],
                        linewidths=0.5, colors='black',zorder=4)

    df_outlier = df[~y_pred]
    df_feedback = df_outlier[(df_outlier["usage proportion"]>df["usage proportion"].median())
                    & (df_outlier["usage_percentage"]>df["usage_percentage"].median())]

    # TRUE PRED
    true_df_feedback = df[df.home.isin(submetered_homes_feedback)]
    print algo_name, len(true_df_feedback)

    # False positives feedback
    df_feedback_fp = df_feedback[~df_feedback.home.isin(submetered_homes_feedback)]

    subplot.scatter(df_feedback_fp["usage proportion"],
                    df_feedback_fp["usage_percentage"], c='gray',alpha=0.6,lw=0.2)

    df_outlier_no_feedback = df_outlier[(df_outlier["usage proportion"]<=df["usage proportion"].median())
                    & (df_outlier["usage_percentage"]<=df["usage_percentage"].median())]

    subplot.scatter(df_outlier_no_feedback["usage proportion"],
                    df_outlier_no_feedback["usage_percentage"], c='gray',alpha=0.6,lw=0.2)

    df_non_outlier = df[y_pred]

    subplot.scatter(df_non_outlier["usage proportion"],
                    df_non_outlier["usage_percentage"], c='gray',alpha=0.6,lw=0.2)


    subplot.scatter(true_df_feedback["usage proportion"],
                    true_df_feedback["usage_percentage"],
                    c='red',alpha=0.6,zorder=5,lw=0.2)


    subplot.axis('tight')

    subplot.set_xlim((0,1))
    subplot.set_ylim((0,100))

    format_axes(ax[i])
    ax[i].axhspan(df["usage_percentage"].median(), df["usage_percentage"].median(), alpha=0.5,lw=0.2)
    ax[i].axvspan(df["usage proportion"].median(), df["usage proportion"].median(), alpha=0.5,lw=0.2)
    ax[i].axhspan(ymin=df["usage_percentage"].median(), ymax=100,xmin=df["usage proportion"].median(),facecolor='green',edgecolor='green',alpha=0.07)
    ax[i].axvspan(xmin=df["usage proportion"].median(), xmax=1,ymin=df["usage_percentage"].median(),facecolor='green',edgecolor='green',alpha=0.07)

    ax[i].set_xlabel("Proportion of usage cycles")
    #ax[i].set_title(algo_name)


ax[0].set_ylabel(r"Usage energy $\%$")
ax[0].set_title("Submetered\n 13 homes will get feedback")
ax[1].set_title("CO\n \#FN = 9, \#FP = 13")
ax[2].set_title("FHMM\n \#FN= 8, \#FP=12")
ax[3].set_title("Hart\n \#FN=7, \#FP=7")


plt.tight_layout()
plt.savefig("../../figures/fridge/disag_usage_energy_ratio.png",bbox_inches="tight")
plt.savefig("../../figures/fridge/disag_usage_energy_ratio.pdf",bbox_inches="tight")
#plt.show()



e = df[df.usage_percentage.isin(XY[-n_outliers:, 1])]
feedback_homes = e[e.usage_percentage>df.usage_percentage.median()]["home"].values

