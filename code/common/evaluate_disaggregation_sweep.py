import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, mean_absolute_error, mean_squared_error
import os
import matplotlib.pyplot as plt
import sys
import matplotlib.patches as mpatches

appliance = sys.argv[1]

RESULTS_PATH = os.path.expanduser("~/git/nilm-actionable/code/bash_runs_%s" %appliance)
script_path = os.path.dirname(os.path.realpath(__file__))

import sys
sys.path.append("../common")
import glob

from common_functions import latexify, format_axes

def get_immediate_subdirectories(dir):
    return [name for name in os.listdir(dir)
            if os.path.isdir(os.path.join(dir, name))]


def f_score(df, algo_name,lim=20):
    gt = (df > lim)[["GT"]]
    pred = (df > 20)[[algo_name]]
    return f1_score(gt["GT"], pred[algo_name])



def mne(df, algo_name):
    out_energy = {}
    x = df[["GT"]]
    gt_energy = x.sum().values[0]
    y = df[[algo_name]]
    algo_energy = y.sum().values[0]
    return np.abs(algo_energy - gt_energy) / gt_energy

def mse(df, algo_name):
    out = {}
    x = df[["GT"]].values
    y = df[[algo_name]].values
    return mean_squared_error(x, y)

def mae(df, algo_name):
    out = {}
    x = df[["GT"]].values
    y = df[[algo_name]].values
    return mean_absolute_error(x, y)


metrics = {"rmse power": mse,
           "error energy": mne,
           "f_score": f_score}


#metrics = {"mae power": mae,
#           "error energy": mne,
#           "f_score": f_score}

import json


def write_results_to_json(out, json_path):
    with open(json_path, "w") as f:
        json.dump(out, f)


def load_results_from_json(json_path):
    with open(json_path, "r") as f:
        out = json.load(f, parse_int=int)
    return out

def results_dictionary():
    subdirs = get_immediate_subdirectories(RESULTS_PATH)
    print(subdirs)
    out = {}

    for dir in subdirs:
        try:
            params = dir.split("_")
            num_states = int(params[0][1:])
            K = int(params[1][1:])
            train_fraction = int(params[2][1:])
            algo_name = params[3]
            if num_states not in out:
                out[num_states] = {}
            if K not in out[num_states]:
                out[num_states][K] = {}
            if train_fraction not in out[num_states][K]:
                out[num_states][K][train_fraction] = {}
            # Find all H5 files
            dir_full_path = os.path.join(RESULTS_PATH, dir)
            homes = glob.glob(dir_full_path + "/*.h5")
            for home in homes:
                print num_states, K, train_fraction, home, algo_name
                home_full_path = os.path.join(os.path.join(RESULTS_PATH, dir), home)
                with pd.HDFStore(home) as store:
                    df = store['/disag'].dropna()
                home_name = home.split("/")[-1].split(".")[0]
                if home_name not in out[num_states][K][train_fraction]:
                    out[num_states][K][train_fraction][home_name] = {}

                for metric_name, metric_func in metrics.iteritems():
                    if metric_name not in out[num_states][K][train_fraction][home_name]:
                        out[num_states][K][train_fraction][home_name][metric_name] = {}
                    out[num_states][K][train_fraction][home_name][metric_name][algo_name] = metric_func(df, algo_name)
        except Exception, e:
            import traceback
            traceback.print_exc()

    return out

def variation_in_num_states(out, K=5, train_fraction=50):

    o = {}

    for metric in ["f_score", "error energy", "rmse power"]:

        o[metric] = {}
        for num_states in out.keys():
            o[metric][num_states] = {}
            for algo in ["FHMM", "Hart", "CO"]:
                o[metric][num_states][algo] = []    
    for num_states in out.keys():
        if K not in out[num_states]:
            K = str(K)
        if train_fraction not in out[num_states][K]:
            train_fraction = str(train_fraction)
        temp = out[num_states][K][train_fraction]
        for home, home_results in temp.iteritems():
            for metric, metric_results in home_results.iteritems():
                for algo, val in metric_results.iteritems():
                    o[metric][num_states][algo].append(val)

    for metric in o.keys():
        for num_states in o[metric].keys():
            for algo in o[metric][num_states].keys():
                o[metric][num_states][algo] = np.median(o[metric][num_states][algo])
    return o

    for num_states in out.keys():
        for metric, metric_results in home_results.iteritems():
            for algo, val in metric_results.iteritems():
                o[metric][num_states][algo] = np.mean(o[metric][num_states][algo])
    return o

def plot_overall(out):
    latexify(columns=2, fig_height=3.9)
    fig, ax = plt.subplots(nrows=3, ncols=3, sharex=True)
    output = {}
    for N in range(2,5):
        output[N] = variation_in_top_k(out, num_states=N)

        for row, metric in enumerate(["f_score", "error energy", "mae power"]):
            df = pd.DataFrame(output[N][metric]).T
            if metric=="error energy":
                df=df.mul(100)


            df[["CO", "FHMM"]].plot(kind="bar", ax=ax[row, N-2], rot=0, legend=False)
            ax[row, N-2].axhline(y=df["Hart"].median(), linestyle="-",color='red', label="Hart")

            format_axes(ax[row, N-2])
    ylims = {"hvac":
                 {
                     0: (0, 1.1),
                     1: (0, 40),
                     2: (0, 500)
                 },
        "fridge":
            {
                0:(0, 0.8),
                1:(0, 140),
                2:(0, 140)
            }
    }
    for i in range(3):
        ax[0,i].set_ylim(ylims[appliance][0])
    for i in range(3):
        ax[1,i].set_ylim(ylims[appliance][1])
    for i in range(3):
        ax[2,i].set_ylim(ylims[appliance][2])
    for i in range(3):
        ax[-1,i].set_xlabel("Top-K")
        ax[0,i].set_title("N=%d" %(i+2))

    ax[0,0].set_ylabel("F score\n (Higher is better)")
    ax[1,0].set_ylabel(r"\% Error" "\n" "in Energy" "\n" "(Lower is better)")
    ax[2,0].set_ylabel("Mean absolute error\n in power (W)\n (Lower is better)")

    co_patch = mpatches.Patch(color='blue', label='CO')
    fhmm_patch =  mpatches.Patch(color='green', label='FHMM')
    hart_patch = mpatches.Patch(color='red', label='Hart',)
    fig.tight_layout()
    fig.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05),
          ncol=4,handles=[co_patch, fhmm_patch, hart_patch],
           labels=["CO","FHMM","Hart"])

    plt.savefig(os.path.join(script_path, "../../figures/%s/%s_accuracy_nilm.pdf" %(appliance, appliance) ),
                bbox_inches="tight")








def plot_variation_num_states(o):
    latexify(fig_height=3.6)
    fig, ax = plt.subplots(nrows=3, sharex=True)
    error_energy = pd.DataFrame(o["error energy"]).T
    error_energy["Hart"] = error_energy["Hart"].value_counts().head(1).index[0]
    error_energy.plot(ax=ax[0], kind="bar", rot=0)
    ax[0].set_ylabel(r"\% Error" "\n" "in Energy")
    mae_power = pd.DataFrame(o["mae power"]).T
    mae_power["Hart"] =  mae_power["Hart"].value_counts().head(1).index[0]
    mae_power.plot(ax=ax[1], kind="bar", legend=False, rot=0)
    ax[1].set_ylabel("Mean absolute error\n in power (W)")
    f_score = pd.DataFrame(o["f_score"]).T
    f_score["Hart"] =  f_score["Hart"].value_counts().head(1).index[0]

    f_score.plot(ax=ax[2], kind="bar", legend=False, rot=0)
    ax[2].set_ylabel(r"F-score")
    ax[2].set_xlabel("Number of states used for modelling")

    for a in ax:
        format_axes(a)
    plt.tight_layout()
    ax[0].legend(loc='upper center', bbox_to_anchor=(0.5, 1.25),
          ncol=3)
    plt.savefig("../../figures/%s/num_states_%s.png" %(appliance, appliance), bbox_inches="tight")
    plt.savefig("../../figures/%s/num_states_%s.pdf" %(appliance, appliance), bbox_inches="tight")


def plot_variation_top_k(o):
    latexify(fig_height=3.6)
    fig, ax = plt.subplots(nrows=3, sharex=True)
    error_energy = pd.DataFrame(o["error energy"]).T
    error_energy["Hart"] = error_energy["Hart"].value_counts().head(1).index[0]
    error_energy.plot(ax=ax[0], kind="bar", rot=0)
    ax[0].set_ylabel(r"\% Error" "\n" "in Energy")
    mae_power = pd.DataFrame(o["mae power"]).T
    mae_power["Hart"] =  mae_power["Hart"].value_counts().head(1).index[0]
    mae_power.plot(ax=ax[1], kind="bar", legend=False, rot=0)
    ax[1].set_ylabel("Mean absolute error\n in power (W)")
    f_score = pd.DataFrame(o["f_score"]).T
    f_score["Hart"] =  f_score["Hart"].value_counts().head(1).index[0]

    f_score.plot(ax=ax[2], kind="bar", legend=False, rot=0)
    ax[2].set_ylabel(r"F-score")

    ax[2].set_xlabel("Top $k$ appliances used for modelling")

    for a in ax:
        format_axes(a)
    plt.tight_layout()
    ax[0].legend(loc='upper center', bbox_to_anchor=(0.5, 1.25),
          ncol=3)
    plt.savefig("../../figures/%s/top_k_%s.png" %(appliance, appliance), bbox_inches="tight")
    plt.savefig("../../figures/%s/top_k_%s.pdf" %(appliance, appliance), bbox_inches="tight")



def variation_in_top_k(out, num_states=2, train_fraction=50):
    o = {}

    for metric in ["f_score", "error energy", "mae power"]:

        o[metric] = {}
        if num_states not in out.keys():
            num_states = str(num_states)
        for k in out[num_states].keys():
            o[metric][k] = {}
            for algo in ["FHMM", "Hart", "CO"]:
                o[metric][k][algo] = []
    for k in out[num_states].keys():
        if train_fraction not in out[num_states][k]:
            train_fraction = str(train_fraction)
        temp = out[num_states][k][train_fraction]
        for home, home_results in temp.iteritems():
            for metric, metric_results in home_results.iteritems():
                for algo, val in metric_results.iteritems():
                    o[metric][k][algo].append(val)

    for k in out[num_states].keys():
        for metric, metric_results in home_results.iteritems():
            for algo, val in metric_results.iteritems():
                o[metric][k][algo] = np.median(o[metric][k][algo])
    return o
