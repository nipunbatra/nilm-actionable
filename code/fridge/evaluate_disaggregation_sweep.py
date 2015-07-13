import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, mean_absolute_error
import os
import matplotlib.pyplot as plt

RESULTS_PATH = os.path.expanduser("~/git/nilm-actionable/code/bash_runs")

import sys
sys.path.append("../common")
import glob

from common_functions import latexify, format_axes

def get_immediate_subdirectories(dir):
    return [name for name in os.listdir(dir)
            if os.path.isdir(os.path.join(dir, name))]


def f_score(df, lim=20):
    gt = (df > lim)[["GT"]]
    pred = (df > 20)[["CO", "FHMM", "Hart"]]
    o = {}
    for algo in ["CO", "FHMM", "Hart"]:
        o[algo] = f1_score(gt["GT"], pred[algo])
    return o


def mne(df):
    out_energy = {}
    x = df[["GT"]]
    gt_energy = x.sum().values[0]
    for algo in ["CO", "FHMM", "Hart"]:
        y = df[[algo]]
        algo_energy = y.sum().values[0]
        out_energy[algo] = np.abs(algo_energy - gt_energy) / gt_energy
    return out_energy


def mae(df):
    out = {}
    x = df[["GT"]].values
    for algo in ["CO", "FHMM", "Hart"]:
        y = df[[algo]].values
        out[algo] = mean_absolute_error(x, y)
    return out


metrics = {"mae power": mae,
           "error energy": mne,
           "f_score": f_score}


def load_results_from_json(json_path):
    import json
    with open(json_path, "r") as f:
        out = json.load(f, parse_int=int)
    return out

def results_dictionary():
    subdirs = get_immediate_subdirectories(RESULTS_PATH)

    out = {}
    for dir in subdirs:
        params = dir.split("_")
        num_states = int(params[0][1:])
        K = int(params[1][1:])
        train_fraction = int(params[2][1:])
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
            print num_states, K, train_fraction, home
            home_full_path = os.path.join(os.path.join(RESULTS_PATH, dir), home)
            with pd.HDFStore(home) as store:
                df = store['/disag'].dropna()
            home_name = home.split("/")[-1].split(".")[0]
            out[num_states][K][train_fraction][home_name] = {}
            for metric_name, metric_func in metrics.iteritems():
                out[num_states][K][train_fraction][home_name][metric_name] = metric_func(df)
    return out

def variation_in_num_states(out, K=5, train_fraction=50):
    o = {}

    for metric in ["f_score", "error energy", "mae power"]:

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

    for num_states in out.keys():
        for metric, metric_results in home_results.iteritems():
            for algo, val in metric_results.iteritems():
                o[metric][num_states][algo] = np.mean(o[metric][num_states][algo])
    return o

def plot_variation_num_states(o):
    latexify(fig_height=3.9)
    fig, ax = plt.subplots(nrows=3, sharex=True)
    error_energy = pd.DataFrame(o["error energy"]).T
    error_energy.plot(ax=ax[0], kind="bar", rot=0)
    ax[0].set_ylabel(r"\% Error" "\n" "in Energy")
    mae_power = pd.DataFrame(o["mae power"]).T
    mae_power.plot(ax=ax[1], kind="bar", legend=False, rot=0)
    ax[1].set_ylabel("Mean absolute error\n in power (W)")
    f_score = pd.DataFrame(o["f_score"]).T
    f_score.plot(ax=ax[2], kind="bar", legend=False, rot=0)
    ax[2].set_ylabel(r"F-score")
    ax[2].set_xlabel("Number of states used for modelling")

    for a in ax:
        format_axes(a)
    plt.tight_layout()
    ax[0].legend(loc='upper center', bbox_to_anchor=(0.5, 1.25),
          ncol=3)
    plt.savefig("../../figures/fridge/num_states_fridge.png", bbox_inches="tight")
    plt.savefig("../../figures/fridge/num_states_fridge.pdf", bbox_inches="tight")



def variation_in_top_k(out, num_states=2, train_fraction=50):
    o = {}

    for metric in ["f_score", "error energy", "mae power"]:

        o[metric] = {}
        for k in out[num_states].keys():
            o[metric][k] = {}
            for algo in ["FHMM", "Hart", "CO"]:
                o[metric][k][algo] = []
    for k in out[num_states].keys():
        temp = out[num_states][k][train_fraction]
        for home, home_results in temp.iteritems():
            for metric, metric_results in home_results.iteritems():
                for algo, val in metric_results.iteritems():
                    o[metric][k][algo].append(val)

    for k in out[num_states].keys():
        for metric, metric_results in home_results.iteritems():
            for algo, val in metric_results.iteritems():
                o[metric][k][algo] = np.mean(o[metric][k][algo])
    return o
