import pandas as pd
import sys
import matplotlib.pyplot as plt

sys.path.append("../common")

from common_functions import latexify, format_axes
dp = pd.read_csv("../../data/fridge/dp_index.csv")

# Plotting histogram of defrost usage
latexify()
df = dp.artifical_sum*100/dp.total
print (df>85).sum(), "Greater than 85"
print (df>90).sum(), "Greater than 90"
ax = df.hist(color='k')
plt.xlabel("Percentage of fridge energy recovered")
plt.ylabel("Number of homes")
format_axes(ax)
plt.grid(False)
plt.tight_layout()
plt.savefig("../../figures/fridge/fridge_energy_recovered.png")
plt.savefig("../../figures/fridge/fridge_energy_recovered.pdf")

# Energy wasted




