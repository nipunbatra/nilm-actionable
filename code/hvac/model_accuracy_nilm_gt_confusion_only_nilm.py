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


latexify(columns=2, fig_height=1.2)
fig, ax = plt.subplots(ncols=3, sharey=True)


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
                "N2_K3_T50_CO",
               "N2_K3_T50_FHMM",
                "N2_K3_T50_Hart"
               ]
cbar_ax = fig.add_axes([.91, .2, .01, .6])


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

    numeric_cols = f
    df[numeric_cols] = df[numeric_cols].div(df[numeric_cols].max())
    accur = accuracy_multiclass(true_labels, pred_labels)

    print folder
    print accur
    print pd.value_counts(pred_labels)
    confusion_df = pd.DataFrame(confusion_matrix(true_labels, pred_labels),
                                index=["Feedback", "No Feedback"],columns=["Feedback", "No Feedback"])
    sns.heatmap(confusion_df, annot=True, fmt="d", linewidths=.5,ax=ax[plot_num%3],cbar=plot_num == 0,
                cbar_ax=None if plot_num else cbar_ax)
    ax[plot_num%3].set_title(return_name(folder)[0])
    #ax[plot_num%3].set_xlabel("Predicted labels")
ax[0].set_ylabel("True labels")

fig.tight_layout(rect=[0, 0, .9, 1])
fig.text(0.5, 0.0, 'Predicted labels', ha='center')

plt.savefig(os.path.expanduser("~/git/nilm-actionable/figures/hvac/hvac_feedback_confusion.pdf"),bbox_inches="tight")
plt.show()

"""
    a = confusion_matrix(true_labels, pred_labels)

    precision, recall, overall_accuracy =find_p_r_a(a)
    algo, N, K = return_name(folder)
    for metric in ["Precision", "Recall", "Accuracy"]:
        if metric not in output:
         output[metric] = {}

    for metric in ["Precision", "Recall", "Accuracy"]:
        if N not in output[metric]:
                output[metric][N] = {}

    for metric in ["Precision", "Recall", "Accuracy"]:
        if K not in output[metric][N]:
            output[metric][N][K] = {}
    o_dict= {"Precision": precision, "Recall": recall, "Accuracy": overall_accuracy}

    for metric in ["Precision", "Recall", "Accuracy"]:
        output[metric][N][K][algo] = o_dict[metric]


gt = {}
hart = {}
gt["Precision"], gt["Recall"], gt["Accuracy"] = find_p_r_a(gt_confusion)
hart["Precision"], hart["Recall"], hart["Accuracy"] = find_p_r_a(hart_confusion)


#latexify(columns=2, fig_height=3.2)
fig, ax = plt.subplots(nrows=3, ncols=3, sharey=True, sharex=True)

for row_num, metric in enumerate(["Precision", "Recall", "Accuracy"]):
    for col_num, num_states in enumerate(['2','3','3']):
        df = pd.DataFrame(output[metric][num_states]).T
        df.plot(kind="bar", ax=ax[row_num, col_num], rot=0, legend=False)
        format_axes(ax[row_num, col_num])
        ax[row_num, col_num].axhline(y=gt[metric], linestyle="-",color='black', label="Submetered")
        ax[row_num, col_num].axhline(y=hart[metric], linestyle="--",color='red', label="Hart")
        ax[-1, col_num].set_xlabel("Top-K")
        ax[0, col_num].set_title("N=%s" %num_states)
    ax[row_num,0].set_ylabel(metric)

co_patch = mpatches.Patch(color='blue', label='CO')
fhmm_patch =  mpatches.Patch(color='green', label='FHMM')
submetered_patch =  mpatches.Patch(color='black', label='Submetered', lw='0.2')
hart_patch = mpatches.Patch(color='red', label='Hart',lw='0.6')

fig.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05),
          ncol=3,handles=[co_patch, fhmm_patch, submetered_patch, hart_patch],
           labels=["CO","FHMM","Submetered","Hart"])

plt.savefig(os.path.join(script_path, "../../figures/hvac/","hvac_feedback_nilm_aggregate.pdf"),
            bbox_inches="tight")
plt.savefig(os.path.join(script_path, "../../figures/hvac/","hvac_feedback_nilm_aggregate.png"),
            bbox_inches="tight")

"""
