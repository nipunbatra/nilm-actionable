import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

import sys

sys.path.append(("../common"))

from common_functions import latexify, format_axes

df = pd.read_csv("../../data/hvac/minutes_GT.csv")

df2 = pd.DataFrame({
    "Work":(df["work_pred"]-df["work_gt"]),
    "Morning":(df["morning_pred"]-df["morning_gt"]),
    "Evening":(df["evening_pred"]-df["evening_gt"]),
    "Night":(df["sleep_pred"]-df["sleep_gt"]),
    })

latexify(fig_height=1.2)
ax = df2.plot(kind="box",color="gray", sym="", vert=False)
plt.xlabel("Predicted setpoint error" "$(^{\circ}$" "F)")
format_axes(ax)
plt.tight_layout()
plt.savefig("../../figures/hvac/model_setpoint.pdf", bbox_inches="tight")
plt.savefig("../../figures/hvac/model_setpoint.png", bbox_inches="tight")