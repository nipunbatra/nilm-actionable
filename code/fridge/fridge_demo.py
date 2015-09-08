import sys

import numpy as np

import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema

sys.path.append("../common")
import os
import matplotlib.patches as mpatches

def read_fridge_csv(csv_path):
    df = pd.read_csv(csv_path, usecols=[1, 2], skiprows=2,
                     index_col=0, names=["timestamp", "power"])
    df.index = pd.to_datetime(df.index)
    return df

import os
df = read_fridge_csv(os.path.expanduser('~Documents/HOBOware/power_42.csv'))['power']
df.index = pd.to_datetime(df.index)
df_res = df['04-8-2015']

from common_functions import latexify, format_axes
latexify(columns=1, fig_height=1.7)


ax = df_res.plot(linewidth=0.7, color='black')
plt.xlabel("Time")
plt.ylabel("Power (W)")
format_axes(ax)

ax.axvspan(pd.to_datetime("04-08-2015 00:00"), pd.to_datetime("04-08-2015 08:00"),facecolor='green',edgecolor='green',alpha=0.3)
ax.axvspan(pd.to_datetime("04-08-2015 09:00"), pd.to_datetime("04-08-2015 10:40"),facecolor='orange',edgecolor='orange',alpha=0.3)
ax.axvspan(pd.to_datetime("04-08-2015 10:50"), pd.to_datetime("04-08-2015 14:55"),facecolor='blue',edgecolor='blue',alpha=0.3)
ax.axvspan(pd.to_datetime("04-08-2015 15:00"), pd.to_datetime("04-08-2015 18:40"),facecolor='red',edgecolor='red',alpha=0.3)

baseline_patch = mpatches.Patch(color='green',alpha=0.3, label='Baseline')
defrost_patch =  mpatches.Patch(color='orange',alpha=0.3, label='Defrost')
defrost_next =  mpatches.Patch(color='blue',alpha=0.3, label='Increased compressor\n runtime due to defrost')
usage = mpatches.Patch(color='red',alpha=0.3, label='Usage')
plt.tight_layout()

plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05),
          ncol=2,handles=[baseline_patch, defrost_patch, defrost_next, usage])

plt.savefig("../../figures/fridge/fridge_illustration.pdf", bbox_inches="tight")
plt.savefig("../../figures/fridge/fridge_illustration.png", bbox_inches="tight")





