import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys

sys.path.append("../common")


BASELINE_DUTY = 0.37

df = pd.read_csv("../../data/fridge/fridge_compressor_door.csv")
df_regular = df[df.Type == "Regular"].copy()


df_regular["on_minutes"] = (pd.to_datetime(df_regular["On end time"]) - pd.to_datetime(df_regular["On start time"])).astype("int") / (1e9 * 60)
df_regular["off_minutes"] = (pd.to_datetime(df_regular["Off end time"]) - pd.to_datetime(df_regular["Off start time"])).astype("int") / (
    1e9 * 60)

df_regular["duty_percentage"] = df_regular.on_minutes / (df_regular.on_minutes + df_regular.off_minutes)

#
energy_activities_total = df_regular["Energy activity"].sum()


precision = {}
recall = {}
for threshold_percentage in range(1, 40):
    threshold = BASELINE_DUTY*1.0*threshold_percentage/100 + BASELINE_DUTY
    v_counts = df_regular[df_regular.duty_percentage>threshold]["Energy activity"].value_counts()
    if False not in v_counts.index:
        false_count = 0
    else:
        false_count = v_counts[False]
    if True not in v_counts.index:
        true_count = 0
    else:
        true_count = v_counts[True]
    precision[threshold_percentage] = true_count*1.0/(true_count+false_count)
    recall[threshold_percentage] = true_count*1.0/energy_activities_total

precision_recall_df = df = pd.DataFrame({"precision":precision, "recall":recall})

from common_functions import latexify, format_axes, SPINE_COLOR

"""
Plotting preision and recall individually

for field in ["precision", "recall"]:
    latexify(columns=1)
    ax = precision_recall_df[field].plot()
    ax.set_ylabel(str.capitalize(field))
    ax.set_xlabel("Percentage threshold")
    format_axes(ax)
    plt.ylim((0, 1))
    plt.tight_layout()
    plt.savefig("../../figures/fridge/%s_fridge_activity.png" %field)
    plt.savefig("../../figures/fridge/%s_fridge_activity.pdf" %field)
    plt.close()
"""

# Plotting precision and recall on a single yy plot

fig = plt.figure()
ax = fig.add_subplot(111)

latexify(columns=1)
#precision_recall_df["precision"].plot(ax=ax)
l1 = ax.plot(precision_recall_df["precision"], 'b-')
ax2 = ax.twinx()
#ax.set_ylabel("Precision")
ax.set_xlabel("Percentage threshold")
l2 = ax2.plot(precision_recall_df["recall"], 'r-')
format_axes(ax)
ax.set_ylim((-0.1, 1.1))
ax2.set_ylim((-0.1, 1.1))

lines = l1+l2
labs = ["Precision", "Recall"]
ax.legend(lines, labs, loc='upper center', bbox_to_anchor=(0.5, 1.25),
          ncol=2, fancybox=True, shadow=True)

for axis in [ax2.xaxis, ax2.yaxis]:
        axis.set_tick_params(direction='out', color=SPINE_COLOR)

#format_axes(ax2)
#plt.legend()
plt.tight_layout()
plt.savefig("../../figures/fridge/precision_recall_fridge_activity.png", bbox_inches="tight")
plt.savefig("../../figures/fridge/precision_recall_fridge_activity.pdf", bbox_inches="tight")
plt.close()



