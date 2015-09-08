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
def find_usage(df):
    cycle_mins =  (df["on_minutes"] + df["off_minutes"])
    usage_df = df.on_minutes - (cycle_mins*BASELINE_DUTY)
    usage_mins = usage_df.sum()
    return usage_mins

non_energy_activities_df = df_regular[~df_regular["Energy activity"]]
energy_activities_df = df_regular[df_regular["Energy activity"]]

true_usage_mins = find_usage(energy_activities_df)

energy_activities_total = df_regular["Energy activity"].sum()


error = {}
precision = {}
recall = {}
for threshold_percentage in range(1, 40):
    threshold = BASELINE_DUTY*1.0*threshold_percentage/100 + BASELINE_DUTY
    pred_df = df_regular[df_regular.duty_percentage>threshold]
    pred_df_mins = find_usage(pred_df)
    error[threshold_percentage] = (pred_df_mins-true_usage_mins)*1.0/true_usage_mins
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




result_df = pd.DataFrame({"error":error, "precision":precision, "recall":recall}).abs()


from common_functions import latexify, format_axes


tableau20 = [(31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),
             (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),
             (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),
             (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),
             (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)]

# Scale the RGB values to the [0, 1] range, which is the format matplotlib accepts.
for i in range(len(tableau20)):
    r, g, b = tableau20[i]
    tableau20[i] = (r / 255., g / 255., b / 255.)

latexify(columns=1, fig_height=1.7)
ax = result_df.plot(colors=[tableau20[0], tableau20[4],tableau20[6]])
ax.set_xlabel(r"Percentage threshold ($P$)")
format_axes(ax)
plt.ylim((-0.1, 1.1))



plt.tight_layout()

L = ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.35),
          ncol=3)
L.get_texts()[0].set_text(r'Usage Energy' '\nProportion')
L.get_texts()[1].set_text(r'Precision')
L.get_texts()[2].set_text(r'Recall')
#plt.tight_layout()


plt.savefig("../../figures/fridge/fridge_activity_energy.png", bbox_inches="tight")
plt.savefig("../../figures/fridge/fridge_activity_energy.pdf", bbox_inches="tight")

"""
for field in ["error"]:
    latexify(columns=1)
    ax = result_df[field].plot()
    ax.set_ylabel(r'Usage Energy Error $\%$')
    ax.set_xlabel(r"Percentage threshold ($P$)")
    format_axes(ax)
    #plt.ylim((0, 1))
    plt.tight_layout()
    plt.savefig("../../figures/fridge/%s_fridge_activity_energy.png" %field)
    plt.savefig("../../figures/fridge/%s_fridge_activity_energy.pdf" %field)
    plt.close()

"""

