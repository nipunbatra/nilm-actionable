import pandas as pd
import sys
sys.path.append("../common")
import matplotlib.pyplot as plt

from common_functions import latexify, format_axes

df = pd.read_csv("../../data/fridge/usage_defrost_cycles.csv")



latexify()
plt.scatter(df["total"].values, df["artifical_sum"].values)

plt.xlabel("Actual fridge aggregate energy (kWh)")
plt.ylabel("Predicted fridge aggregate\n energy (kWh)")
format_axes(plt.gca())

plt.tight_layout()
plt.savefig("../../figures/fridge/breakdown_accuracy.png")
plt.savefig("../../figures/fridge/breakdown_accuracy.pdf")


