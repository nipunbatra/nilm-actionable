import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import scipy.interpolate as inter
import os
script_path = os.path.dirname(os.path.realpath(__file__))

fig, ax = plt.subplots()

from sklearn.externals import joblib

x_num = 5
xpos = np.linspace(0, 1, x_num)

plt.xlim((-0.05,1.05))
for x in xpos:
    plt.axvspan(xmin=x,xmax=x,ymin=0,ymax=1,alpha=0.5)
f = list(('a1', 'a3', 'evening_energy', 'morning_mins'))

df = pd.read_csv("../../data/hvac/minutes_N2_K3_T50_FHMM.csv")
numeric_cols = f
df[numeric_cols] = df[numeric_cols].div(df[numeric_cols].max())

for i, column in enumerate(df[numeric_cols]):
    plt.scatter(xpos[i]*np.ones(len(df)),df[column],alpha=0)
plt.xticks(xpos, numeric_cols)


def name(x):
    if x=="Average" or x=="Good":
        return "Not bad"
    else:
        return x

NUM_CLASSES = 2
df["hvac_class_copy"] = df["hvac_class"].copy()
if NUM_CLASSES == 2:
    df_copy = df.copy()
    df.hvac_class = df.hvac_class.apply(name)
    COLUMN_NAMES = ["Bad", "Not bad"]
else:
    COLUMN_NAMES = ["Average", "Bad", "Good"]

x = xpos
mapping = {"Bad":{"color":"red", "value":0},
           "Not bad":{"color":"blue","value":1}}

np.random.seed(0)
clf = joblib.load(os.path.expanduser("~/git/nilm-actionable/data/hvac/rf_hvac.pkl"))
true_labels = df['hvac_class'].values
pred_labels = clf.predict(df[list(f)])

print pd.value_counts(pred_labels)

for i in range(len(df)):
    y_df = df.ix[i]
    y = df.ix[i][f].values.astype(float)
    y = np.append(y, mapping[pred_labels[i]]["value"])
    s1 = inter.InterpolatedUnivariateSpline (x, y,k=4)
    plt.plot(x, s1(x), color=mapping[y_df["hvac_class"]]["color"],zorder=mapping[y_df["hvac_class"]]["value"]*10)
plt.show()