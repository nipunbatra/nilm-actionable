import pandas as pd
import sys
sys.path.append("../common")
import matplotlib.pyplot as plt

from common_functions import latexify, format_axes

df = pd.read_csv("../../data/fridge/usage_defrost_cycles.csv")

df["usage proportion"] = df["usage_cycles"]/(df["usage_cycles"] + df["non_usage_cycles"])

plt.clf()
latexify()
plt.scatter(df["usage proportion"].values, df["usage_percentage"].values)
plt.axhspan(df["usage_percentage"].median(), df["usage_percentage"].median())
plt.annotate(r'Median usage energy $\%$', xy=(0.05, df["usage_percentage"].median()), xytext=(0.05,15 ),
            arrowprops=dict(facecolor='black', shrink=0.05,  width=0.2, headwidth=1),
            )
plt.axhspan(df["usage_percentage"].median()+10, df["usage_percentage"].median()+10)

plt.annotate(r'10 $\%$ more'"\n"  'usage than median', xy=(0.5, df["usage_percentage"].median()+10), xytext=(0.5, df["usage_percentage"].median()+15),
            arrowprops=dict(facecolor='black', shrink=0.05, width=0.2, headwidth=1),
            )
format_axes(plt.gca())
plt.xlabel("Proportion of cycles affected by usage")
plt.ylabel(r"Usage energy $\%$")
plt.tight_layout()
plt.savefig("../../figures/fridge/usage_energy_ratio.png")
plt.savefig("../../figures/fridge/usage_energy_ratio.pdf")


med = df.usage_percentage.median()
feedback_median = df.usage_percentage[(df.usage_percentage<med+10)&(df.usage_percentage>med)]
feedback_more_than_10 = df.usage_percentage[(df.usage_percentage>med+10)]