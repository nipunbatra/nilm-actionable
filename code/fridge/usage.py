import pandas as pd
import sys
import matplotlib.pyplot as plt

sys.path.append("../common")

from common_functions import latexify, format_axes
dp = pd.read_csv("../../data/fridge/usage_defrost_cycles.csv")

# Plotting histogram of defrost usage
latexify()
ax = dp.usage.hist(color='k')
plt.xlabel("Usage monthly kWh consumption")
plt.ylabel("Number of homes")
format_axes(ax)
plt.grid(False)
plt.tight_layout()
plt.savefig("../../figures/fridge/usage_hist.png")
plt.savefig("../../figures/fridge/usage_hist.pdf")

# Energy wasted
dp[dp.usage>dp.usage.median()].usage - dp.usage.median()
dp_usage_extra_df = dp[dp.usage>dp.usage.median()].copy()
dp_usage_extra_df["extra"] = dp_usage_extra_df["usage"] - dp.usage.median()
q = dp_usage_extra_df.extra/dp_usage_extra_df.total



