from itertools import combinations

import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.externals import joblib
import matplotlib.pyplot as plt
from hvac_ids_to_consider import find_common_dataids
from collections import OrderedDict
f = list(('a1', 'a3', 'evening_energy', 'morning_mins'))
import os
import matplotlib.patches as mpatches
import scipy.interpolate as inter

import sys
sys.path.append("../common")
script_path = os.path.dirname(os.path.realpath(__file__))
import seaborn as sns
sns.set()
from common_functions import latexify, format_axes

NUM_CLASSES = 2

output ={}
gt_confusion = np.array([[25, 9], [8, 16]])
hart_confusion = np.array([[25, 9],[13,11]])


from sklearn.externals import joblib

x_num = 5
xpos = np.linspace(0, 1, x_num)


latexify(columns=2, fig_height=1.8)
fig, ax = plt.subplots(ncols=4, sharey=True)


def powerset(iterable, N_max):
    xs = list(iterable)
    return combinations(xs, N_max)


def accuracy_multiclass(y_true, y_pred):
    confusion = confusion_matrix(y_true, y_pred)
    length = len(confusion)
    out = 0.0
    for i in xrange(length):
        out += confusion[i, i] * 1.0 / np.sum(confusion[i])
    return out / length

def return_name(folder):
    algo_name = folder.split("_")[-1]
    if algo_name=="Hart":
        return "Hart", 2, 3
    elif algo_name=="GT":
        return "Submetered", 2,3
    algo_N = folder.split("_")[0][1]
    algo_K = folder.split("_")[1][1]
    return algo_name, algo_N, algo_K

def find_p_r_a(a):
    precision = 1.0 * a[0][0] / (a[0][0] + a[1][0])
    recall = 1.0 * a[0][0] / (a[0][0] + a[0][1])
    overall_accuracy = 1.0 * (a[0][0] + a[1][1]) / (np.sum(a))
    return precision, recall, overall_accuracy

to_consider = [
                "GT","N4_K3_T50_CO",
               "N2_K3_T50_FHMM",
                "N2_K3_T50_Hart"
               ]
cbar_ax = fig.add_axes([.91, .2, .01, .6])

out = {}
for plot_num, folder in enumerate(to_consider):
    df = pd.read_csv("../../data/hvac/minutes_%s.csv" % folder)
    df["hvac_class_copy"] = df["hvac_class"].copy()
    df = df[df.dataid.isin(find_common_dataids())]
    df.index = range(len(df))

    if NUM_CLASSES == 2:
        df.hvac_class[(df.hvac_class == "Average") | (df.hvac_class == "Good")] = "Not bad"
        COLUMN_NAMES = ["Bad", "Not bad"]
    else:
        COLUMN_NAMES = ["Average", "Bad", "Good"]

    np.random.seed(0)
    clf = joblib.load(os.path.expanduser("~/git/nilm-actionable/data/hvac/rf_hvac.pkl"))
    true_labels = df['hvac_class'].values
    pred_labels = clf.predict(df[list(f)])
    out["True"] = true_labels
    out[return_name(folder)[0]] = pred_labels
out_df = pd.DataFrame(out, index=df.dataid)
