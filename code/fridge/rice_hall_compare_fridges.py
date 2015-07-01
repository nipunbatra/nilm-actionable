import sys

import numpy as np

import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema

sys.path.append("../common")


def read_fridge_csv(csv_path):
    df = pd.read_csv(csv_path, usecols=[1, 2], skiprows=2,
                     index_col=0, names=["timestamp", "power"])
    df.index = pd.to_datetime(df.index)
    return df

paths = {
    5: "../../data/fridge/rice/5.csv",
    4: "../../data/fridge/rice/4.csv",
    3: "../../data/fridge/rice/3.csv",
    2: "../../data/fridge/rice/2.csv"
}

dfs = {}
for fridge_number, fridge_csv_path in paths.iteritems():
    print fridge_number
    dfs[fridge_number] = read_fridge_csv(csv_path=fridge_csv_path)
    dfs[fridge_number] = dfs[fridge_number]["2015-04-12":"2015-04-12 10:00"]

PLOT = False

if PLOT:
    from common_functions import latexify, format_axes
    latexify(columns=1, fig_height=4.4)

    fig, ax = plt.subplots(nrows=4, sharex=True)
    count = 0
    for fridge_number, fridge_df in dfs.iteritems():
        print fridge_number
        fridge_df["power"].plot(ax=ax[count])
        ax[count].set_ylabel("Power (W)")
        ax[count].set_title("Fridge %d" %fridge_number)
        format_axes(ax[count])
        count += 1
    ax[count-1].set_xlabel("Time")
    plt.tight_layout()
    plt.savefig("../../figures/fridge/rice_hall_comparison.pdf")
    plt.savefig("../../figures/fridge/rice_hall_comparison.png")
    plt.show()


#Printing the average steady state and transient power


print "Fridge #, Transient_power, Steady_State_power"
for fridge_number, fridge_df in dfs.iteritems():
    df = fridge_df["power"]

    # Finding transients
    df2 = df.ix[argrelextrema(df.values, np.greater)[0]]
    if fridge_number==2:
        threshold=200
    else:
        threshold=100
    tr = df2[df2>threshold]
    #Finding steady state
    st = df[(df>50)&(df<200)]
    print fridge_number, df.mean(), tr.mean(), st.mean()




