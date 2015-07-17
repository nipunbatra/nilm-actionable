import sys

sys.path.append("../common")
import matplotlib.pyplot as plt
import warnings
from common_functions import latexify, format_axes
from nilmtk import DataSet
import nilmtk
import pandas as pd

warnings.filterwarnings("ignore")

ds = DataSet("/Users/nipunbatra/Downloads/wikienergy-2.h5")
fridges = nilmtk.global_meter_group.select_using_appliances(type='fridge')

df = pd.read_csv("../../data/fridge/usage_defrost_cycles.csv")
df["days"] = df["total_mins"] / 1440.0
df["num_defrost_per_day"] = df["defrost_cycles"] / df["days"]

X = df["num_defrost_per_day"].values
Y = df["defrost_percentage"].values

XY = df[["num_defrost_per_day", "defrost_percentage"]].values

import numpy as np
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
xx, yy = np.meshgrid(np.linspace(X.min() - 0.3, X.max() + 0.1, 500), np.linspace(Y.min() - 5, Y.max() + 5, 500))
n_inliers = int((1. - outliers_fraction) * n_samples)
n_outliers = int(outliers_fraction * n_samples)

latexify()
# Fit the problem with varying cluster separation
np.random.seed(42)
# Data generation


# Fit the model with the One-Class SVM
# plt.figure(figsize=(10, 5))

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
subplot = plt.subplot(1, 1, 1)
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
subplot.set_xlim((X.min(), X.max()))
subplot.set_ylim((Y.min(), Y.max()))
plt.axhspan(df["defrost_percentage"].median(), df["defrost_percentage"].median())
plt.axvspan(df["num_defrost_per_day"].median(), df["num_defrost_per_day"].median())

# plt.subplots_adjust(0.04, 0.1, 0.96, 0.94, 0.1, 0.26)
format_axes(plt.gca())
plt.xlabel("Number of defrost cycles per day")
plt.ylabel(r"Defrost energy $\%$")
ylims = plt.ylim()
plt.ylim((ylims[0] - 5, ylims[1] + 5))
xlims = plt.xlim()
plt.xlim((xlims[0] - 0.3, xlims[1] + 0.1))
plt.tight_layout()

plt.savefig("../../figures/fridge/defrost_energy_cycles.png")
plt.savefig("../../figures/fridge/defrost_energy_cycles.pdf")

e = df[df.defrost_percentage.isin(XY[-n_outliers:, 1])]
feedback_homes = e[e.defrost_percentage>df.defrost_percentage.median()]["home"].values