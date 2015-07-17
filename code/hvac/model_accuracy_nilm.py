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

import sys
sys.path.append("../common")
script_path = os.path.dirname(os.path.realpath(__file__))

from common_functions import latexify, format_axes

NUM_CLASSES = 2

output = OrderedDict()
gt_confusion = np.array([[25, 9], [8, 16]])
output["Submetered"] = {"Precision": 25.0 / 33, "Recall": 25.0 / 34, "Accuracy": 41.0 / 58}


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

def find_name(folder):
    algo_name = folder.split("_")[-1]
    algo_N = folder.split("_")[0][1]
    algo_K = folder.split("_")[1][1]
    if algo_name=="Hart":
        return "Hart"
    else:
        return algo_name+" (N= "+algo_N+", K= "+algo_K+")"


to_consider = ["N2_K3_T50_CO", "N2_K4_T50_CO", "N2_K5_T50_CO",
               "N3_K3_T50_CO", "N3_K4_T50_CO", "N3_K5_T50_CO",
               "N2_K3_T50_FHMM","N2_K4_T50_FHMM","N2_K5_T50_FHMM",
               "N3_K3_T50_FHMM","N3_K4_T50_FHMM","N3_K5_T50_FHMM",
               "N2_K3_T50_Hart" ]

for folder in to_consider:
    df = pd.read_csv("../../data/hvac/minutes_%s.csv" % folder)
    df["hvac_class_copy"] = df["hvac_class"].copy()
    df = df[df.dataid.isin(find_common_dataids())]

    if NUM_CLASSES == 2:
        df.hvac_class[(df.hvac_class == "Average") | (df.hvac_class == "Good")] = "Not bad"
        COLUMN_NAMES = ["Bad", "Not bad"]
    else:
        COLUMN_NAMES = ["Average", "Bad", "Good"]

    np.random.seed(0)
    clf = joblib.load(os.path.expanduser("~/git/nilm-actionable/data/hvac/rf_hvac.pkl"))
    true_labels = df['hvac_class'].values
    pred_labels = clf.predict(df[list(f)])
    accur = accuracy_multiclass(true_labels, pred_labels)

    print folder
    print accur
    print confusion_matrix(true_labels, pred_labels)
    a = confusion_matrix(true_labels, pred_labels)
    precision = 1.0 * a[0][0] / (a[0][0] + a[1][0])
    recall = 1.0 * a[0][0] / (a[0][0] + a[0][1])
    overall_accuracy = 1.0 * (a[0][0] + a[1][1]) / (np.sum(a))

    algo_identifier = find_name(folder)
    output[algo_identifier] = {"Precision": precision, "Recall": recall, "Accuracy": overall_accuracy}

out_df = pd.DataFrame(output)
#out_df.columns = ["Submetered", "CO (N=2, K=3)","CO (N=3, K=3)","CO (N=3, K=4)","FHMM (N=2, K=3)","FHMM (N=3, K=3)", "Hart"]

latexify(columns=2, fig_height=2.6)
ax = out_df.plot(kind="bar", rot=0)
format_axes(ax)
plt.tight_layout()
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.70, box.height])

# Put a legend to the right of the current axis
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.savefig(os.path.join(script_path, "../../figures/hvac/","hvac_feedback_nilm.pdf"))
plt.savefig(os.path.join(script_path, "../../figures/hvac/","hvac_feedback_nilm.png"))