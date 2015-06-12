import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


df = pd.read_csv("fridge_compressor_door.csv")

df["on_minutes"] = (pd.to_datetime(df["On end time"]) - pd.to_datetime(df["On start time"])).astype("int") / (1e9 * 60)
df["off_minutes"] = (pd.to_datetime(df["Off end time"]) - pd.to_datetime(df["Off start time"])).astype("int") / (
    1e9 * 60)

bins = np.arange(df.on_minutes.min(), df.on_minutes.max() + 5, 5)

df_regular = df[df.Type == "Regular"]
df_regular_non_freezer = df_regular[df_regular["Freezer used"] == False]
df_regular_non_freezer_non_hot = df_regular_non_freezer[df_regular_non_freezer["Hot Food"] == False]
df_all_above = df_regular_non_freezer_non_hot[df_regular_non_freezer_non_hot["Fridge opened"] == False]

ax = df.on_minutes.hist(bins=bins, label="All cycles")
df_regular.on_minutes.hist(bins=bins, ax=ax, alpha=1, label="Removed defrost cycles")
df_regular_non_freezer.on_minutes.hist(bins=bins, ax=ax, alpha=1, label="Removed freezer cycles")
df_regular_non_freezer_non_hot.on_minutes.hist(bins=bins, ax=ax, alpha=1, label="Removed hot food cycles")
df_all_above.on_minutes.hist(bins=bins, ax=ax, alpha=1, label="Removed all fridge open cycles")

plt.legend()
plt.xlabel("Fridge compressor on duration (mins)")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("on.png")

""""







"""""

"""
In [107]: %hist
import pandas as pd
df = pd.read_csv("fridge_compressor_door.csv")
df["on_minutes"] = (pd.to_datetime(df["On end time"]) - pd.to_datetime(df["On start time"])).astype("int")/(1e9*60)
df["off_minutes"] = (pd.to_datetime(df["Off end time"]) - pd.to_datetime(df["Off start time"])).astype("int")/(1e9*60)
df_regular = df[df.Type=="Regular"]
%matplotlib qt
df.on_minutes.hist()
df_regular = df[df.Type=="Regular"]
df_regular.on_minutes.hist()
maximum = max(c.on.max(), cf.on.max())
    minimum = min(c.on.min(), cf.on.min())
    bins=np.arange(minimum, maximum + binwidth, binwidth)
import numpy as np
maximum = max(c.on.max(), cf.

)
    bins=np.arange(df.on_minutes.min(), df.on_minutes.max() + 5, 5)
df.on_minutes.hist(bins=bins)
ax = _
df_regular.on_minutes.hist(bins=bins, ax=ax)
df.on_minutes.hist(bins=bins)
ax = _
df_regular.on_minutes.hist(bins=bins, ax=ax, alpha=0.3)
import seaborn as sns
ax=df.on_minutes.hist(bins=bins)
df_regular.on_minutes.hist(bins=bins, ax=ax, alpha=0.3)
ax=df.on_minutes.hist(bins=bins)
df_regular.on_minutes.hist(bins=bins, ax=ax, alpha=0.7)
df.columns
len(df_regular)
df_regular[(df_regular["Max temp delta on"]>0.5)|(df_regular["Max temp delta off"]>0.5)]
df_2 = df_regular[(df_regular["Max temp delta on"]>0.5)|(df_regular["Max temp delta off"]>0.5)]
len(df_2)
df_2 = df_regular[(df_regular["Max temp delta on"]<0.5)|(df_regular["Max temp delta off"]<0.5)]
len(df_2)
df_2 = df_regular[(df_regular["Max temp delta on"]<0.5)&(df_regular["Max temp delta off"]<0.5)]
len(df_2)
df_2.hist(bins=bins, ax=ax, alpha=0.5)
%paste
%paste
df_regular
df_regular.describe()
df_regular_non_freezer
df_regular_non_freezer.describe()
df_regular_non_freezer.head()
df_regular_non_freezer.describe()
df_regular.describe()
ax = df.on_minutes.hist(bins=bins)
df_regular.hist(bins=bins, ax = ax)
ax = df.on_minutes.hist(bins=bins)
df_regular.on_minutes.hist(bins=bins, ax = ax)
plt.legend()
import matplotlib.pyplot as plt
plt.legend()
df_regular_non_freezer.on_minutes.hist(bins=bins, ax = ax)
df_regular_non_freezer_non_hot = df_regular_non_freezer[df_regular_non_freezer["Hot Food"]==False]
df_regular_non_freezer_non_hot.on_minutes.hist(bins=bins, ax = ax)
%hist
%paste
%paste
df
df.off_minutes
df[df.opened]
df.columns
df[df["Fridge opened"]
]
df[df["Fridge opened"]].describe()
df[~df["Fridge opened"]].describe()
df[~df["Fridge opened"]].on_minutes
df[~df["Fridge opened"]].on_minutes.hist()
%paste
%paste
plt.legend()
%paste
%paste
%paste
%paste
%paste
%paste
!open on.png
df_regular_non_freezer_non_hot.on_minutes
df_regular_non_freezer_non_hot.on_minutes.describe()
df_regular_non_freezer_non_hot
df_regular_non_freezer_non_hot[df_regular_non_freezer_non_hot["Fridge opened"]==False]
df_regular_non_freezer_non_hot[df_regular_non_freezer_non_hot["Fridge opened"]==False].on_minutes
df_regular_non_freezer_non_hot[df_regular_non_freezer_non_hot["Fridge opened"]==False].on_minutes.describe()
df_regular_non_freezer_non_hot[df_regular_non_freezer_non_hot["Fridge opened"]==True].on_minutes.describe()
df_regular_non_freezer_non_hot[df_regular_non_freezer_non_hot["Fridge opened"]==True].on_minutes
df_regular_non_freezer_non_hot[df_regular_non_freezer_non_hot["Fridge opened"]==True]
df
df.describe()
df.on_minutes.describe()
df_regular.on_minutes.describe()
df_regular_non_freezer.on_minutes.describe()
df_regular_non_freezer_non_hot.on_minutes.describe()
df_all_above.on_minutes.describe()
df_regular_non_freezer[df_regular_non_freezer["Fridge opened"]]
df_regular_non_freezer[df_regular_non_freezer["Fridge opened"]].on_minutes
df_regular_non_freezer[~df_regular_non_freezer["Fridge opened"]].on_minutes
df_regular_non_freezer[~df_regular_non_freezer["Fridge opened"]].on_minutes.describe()
df_regular_non_freezer[df_regular_non_freezer["Fridge opened"]].on_minutes.describe()
df_regular_non_freezer_non_hot[df_regular_non_freezer_non_hot["Fridge opened"]].on_minutes.describe()
df_regular_non_freezer_non_hot[~df_regular_non_freezer_non_hot["Fridge opened"]].on_minutes.describe()
df_regular_non_freezer_non_hot[df_regular_non_freezer_non_hot["Fridge opened"]].on_minutes
df_regular[df_regular["Hot Food"]].on_minutes.describe()
df_regular_non_freezer[df_regular_non_freezer["Hot food"]].on_minutes
df_regular_non_freezer[df_regular_non_freezer["Hot Food"]].on_minutes
df_regular_non_freezer[df_regular_non_freezer["Hot Food"]].on_minutes.describe()
df_regular_non_freezer[df_regular_non_freezer["Hot Food"]].on_minutes
"""

