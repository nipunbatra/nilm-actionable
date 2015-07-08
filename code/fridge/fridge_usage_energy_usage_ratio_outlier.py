import sys

import pandas as pd

sys.path.append("../common")

from common_functions import latexify, format_axes

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
xx, yy = np.meshgrid(np.linspace(X.min()-0.1, X.max()+0.1, 500), np.linspace(Y.min()-5, Y.max()+5, 500))
n_inliers = int((1. - outliers_fraction) * n_samples)
n_outliers = int(outliers_fraction * n_samples)

plt.clf()
latexify()
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
subplot.set_xlim((X.min(), X.max()))
subplot.set_ylim((Y.min(), Y.max()))
plt.axhspan(df["usage_percentage"].median(), df["usage_percentage"].median())
plt.axvspan(df["usage proportion"].median(), df["usage proportion"].median())

#plt.subplots_adjust(0.04, 0.1, 0.96, 0.94, 0.1, 0.26)
format_axes(plt.gca())
plt.xlabel("Proportion of usage cycles")
plt.ylabel(r"Usage energy $\%$")
ylims = plt.ylim()
plt.ylim((ylims[0]-5, ylims[1]+5))
xlims = plt.xlim()
plt.xlim((xlims[0]-0.1, xlims[1]+0.1))
plt.tight_layout()
#plt.savefig("../../figures/fridge/usage_energy_ratio.png")
#plt.savefig("../../figures/fridge/usage_energy_ratio.pdf")
#plt.show()

e = df[df.usage_percentage.isin(XY[-n_outliers:, 1])]
feedback_homes = e[e.usage_percentage>df.usage_percentage.median()]["home"].values

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