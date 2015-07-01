import pandas as pd
import sys
sys.path.append("../common")
import matplotlib.pyplot as plt
import warnings
from common_functions import latexify, format_axes
from nilmtk import DataSet
import nilmtk
import pandas as pd
import numpy as np
from numba import autojit, jit

warnings.filterwarnings("ignore")

ds = DataSet("/Users/nipunbatra/Downloads/wikienergy-2.h5")
fridges = nilmtk.global_meter_group.select_using_appliances(type='fridge')

df = pd.read_csv("../../data/fridge/usage_defrost_cycles.csv")
df["days"] = df["total_mins"]/1440.0
df["num_defrost_per_day"] = df["defrost_cycles"]/df["days"]

#df["usage proportion"] = df["usage_cycles"]/(df["usage_cycles"] + df["non_usage_cycles"])

latexify(columns=1)
plt.scatter(df["num_defrost_per_day"].values, df["defrost_percentage"].values)
format_axes(plt.gca())
plt.xlabel("Number of defrost cycles per day")
plt.ylabel(r"Defrost energy $\%$")
plt.tight_layout()
plt.savefig("../../figures/fridge/defrost_energy_cycles.png")
plt.savefig("../../figures/fridge/defrost_energy_cycles.pdf")

plt.clf()
fig = plt.figure()
ax = fig.add_subplot(111)
feedback = df[df["num_defrost_per_day"]>1]
median_defrost_num_per_day = df["num_defrost_per_day"].median()
energy_saving_potential = (feedback["num_defrost_per_day"]-median_defrost_num_per_day)*feedback["defrost_percentage"]/feedback["num_defrost_per_day"]
energy_saving_potential.name = ""
ax = energy_saving_potential.plot(kind='box', ax=ax)
ax.set_xlabel("")
ax.set_ylabel(r"$\%$ Energy savings")
format_axes(ax)
plt.savefig("../../figures/fridge/defrost_energy_saving.png")
plt.savefig("../../figures/fridge/defrost_energy_saving.pdf")

# PLotting the fridge with unusually high defrost
home = int(feedback.ix[feedback.num_defrost_per_day.argmax()]["home"])
f = fridges.meters[home].load().next()[('power','active')]


plt.clf()
fig = plt.figure()
ax = fig.add_subplot(111)
f.head(1440).plot(ax=ax)
format_axes(ax)
plt.xlabel("Time")
plt.ylabel("Power consumption (W)")
plt.tight_layout()
plt.savefig("../../figures/fridge/defrost_high.png")
plt.savefig("../../figures/fridge/defrost_high.pdf")
"""
med = df.usage_percentage.median()
feedback_median = df.usage_percentage[(df.usage_percentage<med+10)&(df.usage_percentage>med)]
feedback_more_than_10 = df.usage_percentage[(df.usage_percentage>med+10)]
"""