import sys

import pandas as pd

sys.path.append("../common")

from common_functions import latexify, format_axes

algos = ["CO", "FHMM", "Hart"]


df = pd.read_csv("../../data/fridge/usage_defrost_cycles.csv")

df["usage proportion"] = df["usage_cycles"]/(df["usage_cycles"] + df["non_usage_cycles"])

X = df["usage proportion"].values
Y = df["usage_percentage"].values

XY = df[["usage proportion","usage_percentage" ]].values


import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

from sklearn.covariance import EllipticEnvelope

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

latexify(fig_height=1.5)
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
subplot = plt.subplot(1, 1,  1)

a = subplot.contour(xx, yy, Z, levels=[threshold],
                        linewidths=0.5, colors='black',zorder=4)

df_outlier = df[~y_pred]
df_feedback = df_outlier[(df_outlier["usage proportion"]>df["usage proportion"].median())
                    & (df_outlier["usage_percentage"]>df["usage_percentage"].median())]

df_regular = df[y_pred]

df_outlier_no_feedback = df_outlier[(df_outlier["usage proportion"]<=df["usage proportion"].median())
                    | (df_outlier["usage_percentage"]<=df["usage_percentage"].median())]
subplot.scatter(df_regular["usage proportion"],
                    df_regular["usage_percentage"],
                    c='gray',alpha=0.6,zorder=0,lw=0.2)

subplot.scatter(df_outlier_no_feedback["usage proportion"],
                    df_outlier_no_feedback["usage_percentage"],
                    c='gray',alpha=0.6,zorder=0,lw=0.2)

subplot.scatter(df_feedback["usage proportion"],
                    df_feedback["usage_percentage"],
                    c='red',alpha=0.6,zorder=5,lw=0.2)
subplot.axis('tight')

subplot.set_xlim((0,1))
subplot.set_ylim((0,40))
med_x = df["usage proportion"].median()
plt.axhspan(df["usage_percentage"].median(), df["usage_percentage"].median(),alpha=0.5,lw=0.2)
plt.axvspan(med_x, df["usage proportion"].median(),alpha=0.5,lw=0.2)
plt.axhspan(ymin=df["usage_percentage"].median(), ymax=40,xmin=med_x,facecolor='green',edgecolor='green',alpha=0.07)
#plt.axhspan(ymin=df["usage_percentage"].median(), ymax=40,xmin=med_x,facecolor='green',edgecolor='green',alpha=0.07)
plt.axvspan(xmin=df["usage proportion"].median(), xmax=1,ymin=df["usage_percentage"].median(),ymax=40,facecolor='green',edgecolor='green',alpha=0.07)
# plt.subplots_adjust(0.04, 0.1, 0.96, 0.94, 0.1, 0.26)
format_axes(plt.gca())
plt.xlabel("Proportion of usage cycles")
plt.ylabel(r"Usage energy $\%$")
plt.tight_layout()
plt.savefig("../../figures/fridge/usage_energy_ratio.png",bbox_inches="tight")
plt.savefig("../../figures/fridge/usage_energy_ratio.pdf",bbox_inches="tight")
#plt.show()



e = df[df.usage_percentage.isin(XY[-n_outliers:, 1])]
feedback_homes = e[e.usage_percentage>df.usage_percentage.median()]["home"].values

