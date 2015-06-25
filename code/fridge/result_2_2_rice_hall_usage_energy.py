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

energy_activities_df = df_regular[df_regular["Energy activity"]]

true_usage_mins = find_usage(energy_activities_df)


error = {}
threshold_percentage = 11
threshold = BASELINE_DUTY*1.0*threshold_percentage/100 + BASELINE_DUTY
pred_df = df_regular[df_regular.duty_percentage>threshold]
pred_df_mins = find_usage(pred_df)
for error_power in range(-20, 20, 2):
    pred_power_mins = pred_df_mins*(1+error_power/100.0)

    error[error_power] = (pred_power_mins-true_usage_mins)*1.0/true_usage_mins




result_df = pd.DataFrame({"error":error})


from common_functions import latexify, format_axes

for field in ["error"]:
    latexify(columns=1)
    ax = result_df[field].plot(kind='bar')
    ax.set_ylabel(str.capitalize(field))
    ax.set_xlabel("Percentage error in power \n prediction by NILM algorithm")
    format_axes(ax)
    #plt.ylim((, 1))
    plt.tight_layout()
    plt.savefig("../../figures/fridge/%s_fridge_activity_energy_sensitivity.png" %field)
    plt.savefig("../../figures/fridge/%s_fridge_activity_energy_sensitivity.pdf" %field)
    plt.close()



