import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

from sklearn.covariance import EllipticEnvelope
sys.path.append("../common")

from common_functions import latexify, format_axes

script_path = os.path.dirname(os.path.realpath(__file__))

DATA_PATH = os.path.join(script_path, "..","..","data/fridge")

submetered_homes_feedback = np.array([142, 144, 146, 151, 152, 155, 157, 159, 163, 167, 169, 170])

latexify(columns=2, fig_height=2.8)
algo_total = ["N2_K3_CO","N2_K4_CO","N2_K3_FHMM","N2_K4_FHMM", "N2_K3_Hart"]
#algo_total = ["N2_K3_CO","N2_K4_CO","N2_K3_FHMM","N2_K4_FHMM", "N2_K3_Hart"]
ncols = len(algo_total)
fig, ax = plt.subplots(ncols=3, sharey=True)

for i, algo in enumerate(algo_total):

    df = pd.read_csv(os.path.join(DATA_PATH, "%s_usage_defrost_cycles.csv" %algo)).dropna()

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
    xx, yy = np.meshgrid(np.linspace(-0.1, 1.1, 1000), np.linspace(0, 100, 1000))
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
    subplot.contourf(xx, yy, Z, levels=np.linspace(Z.min(), threshold, 7),
                     cmap=plt.cm.Blues_r)
    a = subplot.contour(xx, yy, Z, levels=[threshold],
                        linewidths=2, colors='red')
    subplot.contourf(xx, yy, Z, levels=[threshold, Z.max()],
                     colors='orange')
    b = subplot.scatter(XY[:-n_outliers, 0], XY[:-n_outliers, 1], c='white')
    c = subplot.scatter(XY[-n_outliers:, 0], XY[-n_outliers:, 1], c='white')
    subplot.axis('tight')
    subplot.legend(
        [a.collections[0]],
        ['Learned decision function'], loc=4)
    #subplot.set_xlabel("%d. %s (errors: %d)" % (1, "rob", n_errors))
    subplot.set_xlim((-0.1, 1.1))
    subplot.set_ylim((0, 100))
    ax[i].axhspan(df["usage_percentage"].median(), df["usage_percentage"].median())
    ax[i].axvspan(df["usage proportion"].median(), df["usage proportion"].median())

    #plt.subplots_adjust(0.04, 0.1, 0.96, 0.94, 0.1, 0.26)
    format_axes(ax[i])
    ax[i].set_xlabel("Proportion of usage cycles")

    #ylims = plt.ylim()
    #plt.ylim((ylims[0]-5, ylims[1]+5))
    #xlims = plt.xlim()
    #plt.xlim((xlims[0]-0.1, xlims[1]+0.1))



    e = df[df.usage_percentage.isin(XY[-n_outliers:, 1])]
    feedback_homes = e[e.usage_percentage>df.usage_percentage.median()]["home"].values
    print "Predicted:", feedback_homes
    extra_pred = np.setdiff1d(feedback_homes, submetered_homes_feedback)
    missed = np.setdiff1d(submetered_homes_feedback, feedback_homes)
    print "Extra predicted:", extra_pred
    print "Missed:", missed
    title_string = algo+"\n"+"%d out of %d homes extra predicted\n%d out of %d homes missed" %(len(extra_pred), 84, len(missed), 12)
    ax[i].set_title(title_string)
ax[0].set_ylabel(r"Usage energy $\%$")
plt.tight_layout()

plt.savefig("../../figures/fridge/disag_usage_energy_ratio.png")
plt.savefig("../../figures/fridge/disag_usage_energy_ratio.pdf")
plt.show()
"""
plt.clf()
latexify()
plt.scatter(df["usage proportion"].values, df["usage_percentage"].values)
plt.axhspan(df["usage_percentage"].median(), df["usage_percentage"].median())
plt.annotate(r'Median usage energy $\%$', xy=(0.05, df["usage_percentage"].median()), xytext=(0.05,15 ),
            arrowprops=dict(facecolor='black', shrink=0.05,  width=0.2, headwidth=1),
            )
plt.axhspan(df["usage_percentage"].median()+10, df["usage_percentage"].median()+10)

plt.annotate(r'10 $\%$ more'"\n"  'usage than median', xy=(0.5, df["usage_percentage"].median()+10), xytext=(0.5, df["usage_percentage"].median()+15),
            arrowprops=dict(facecolor='black', shrink=0.05, width=0.2, headwidth=1),
            )
format_axes(plt.gca())
plt.xlabel("Proportion of cycles affected by usage")
plt.ylabel(r"Usage energy $\%$")
plt.tight_layout()
plt.savefig("../../figures/fridge/usage_energy_ratio.png")
plt.savefig("../../figures/fridge/usage_energy_ratio.pdf")


med = df.usage_percentage.median()
feedback_median = df.usage_percentage[(df.usage_percentage<med+10)&(df.usage_percentage>med)]
feedback_more_than_10 = df.usage_percentage[(df.usage_percentage>med+10)]
"""