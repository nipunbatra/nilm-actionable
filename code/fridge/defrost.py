import pandas as pd
import sys
import matplotlib.pyplot as plt

sys.path.append("../common")

from common_functions import latexify, format_axes
dp = pd.read_csv("../../data/fridge/usage_defrost_cycles.csv")
# Plotting histogram of defrost usage
latexify()
ax = dp.defrost.hist(color='k')
plt.xlabel("Defrost monthly kWh consumption")
plt.ylabel("Number of homes")
format_axes(ax)
plt.grid(False)
plt.tight_layout()
plt.savefig("../../figures/fridge/defrost_usage_hist.png")
plt.savefig("../../figures/fridge/defrost_usage_hist.pdf")

# Energy wasted
dp_defrost_extra_df = dp[dp.defrost>2*dp.defrost.median()].copy()
dp_defrost_extra_df["extra"] = dp_defrost_extra_df["defrost"] - dp.defrost.median()
dp_defrost_extra_df.extra*100/dp_defrost_extra_df.total



