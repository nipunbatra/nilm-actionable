import pandas as pd
import matplotlib.pyplot as plt
import os

import sys
sys.path.append("../common")

from common_functions import latexify, format_axes
latexify()
df = pd.read_csv("../../data/fridge/usage_defrost_cycles.csv")

splits = ["baseline","usage","defrost"]
column_names = [x+"_percentage" for x in splits]

df = df[column_names]
df = df.rename(columns={k:v for k,v in zip(column_names, splits)})

ax = df.plot(kind="box")
format_axes(ax)

plt.ylabel("Percentage contribution")
plt.savefig("../../figures/fridge/box.pdf", bbox_inches="tight")
plt.savefig("../../figures/fridge/box.png", bbox_inches="tight")