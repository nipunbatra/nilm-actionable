import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


BASELINE_DUTY = 0.37

df = pd.read_csv("../../data/fridge/fridge_compressor_door.csv")
df_regular = df[df.Type == "Regular"].copy()


df_regular["on_minutes"] = (pd.to_datetime(df_regular["On end time"]) - pd.to_datetime(df_regular["On start time"])).astype("int") / (1e9 * 60)
df_regular["off_minutes"] = (pd.to_datetime(df_regular["Off end time"]) - pd.to_datetime(df_regular["Off start time"])).astype("int") / (
    1e9 * 60)

df_regular["duty_percentage"] = df_regular.on_minutes / (df_regular.on_minutes + df_regular.off_minutes)

#
energy_activities_total = df_regular["Energy activity"].sum()


precision = {}
recall = {}
for threshold_percentage in range(1, 20):
    threshold = BASELINE_DUTY*1.0*threshold_percentage/100 + BASELINE_DUTY
    v_counts = df_regular[df_regular.duty_percentage>threshold]["Energy activity"].value_counts()
    if False not in v_counts.index:
        false_count = 0
    else:
        false_count = v_counts[False]
    if True not in v_counts.index:
        true_count = 0
    else:
        true_count = v_counts[True]
    precision[threshold_percentage] = true_count*1.0/(true_count+false_count)
    recall[threshold_percentage] = true_count*1.0/energy_activities_total

precision_recall_df = df = pd.DataFrame({"precision":precision, "recall":recall})


# Inherent variation in the duty percentage when neither of the previous 3 cycles had any consumption
inherent_variation = df_regular[~df_regular["Last 3 cycles"]].duty_percentage
"""
In [62]: df_regular[~df_regular["Last 3 cycles"]].duty_percentage
Out[62]:
0     0.373825
1     0.371013
2     0.378049
3     0.372217
4     0.376917
5     0.377868
6     0.376465
7     0.362930
27    0.376630
28    0.377926
29    0.369412
30    0.373795
44    0.400810
45    0.374324
46    0.430704
47    0.381484
48    0.385373
49    0.379256
50    0.380792
51    0.375395
52    0.383773
53    0.375875
54    0.370883
55    0.367636
Name: duty_percentage, dtype: float64

"""

# Duty percentage when either of the previous 3 cycles had activity
"""
In [63]: df_regular[df_regular["Last 3 cycles"]].duty_percentage
Out[63]:
8     0.397442
14    0.556110
15    0.483714
16    0.493743
17    0.480300
18    0.461945
19    0.374091
20    0.407882
21    0.371659
22    0.485552
23    0.277860
24    0.397912
25    0.383614
26    0.386503
31    0.375938
32    0.394750
38    0.508435
39    0.536125
40    0.554333
41    0.512517
42    0.562844
43    0.497526
56    0.374641
57    0.483445
Name: duty_percentage, dtype: float64
"""




df_regular_non_freezer = df_regular[df_regular["Freezer used"] == False]
df_regular_non_freezer_non_hot = df_regular_non_freezer[df_regular_non_freezer["Hot Food"] == False]
df_all_above = df_regular_non_freezer_non_hot[df_regular_non_freezer_non_hot["Fridge opened"] == False]


# ax = df.duty_percentage.hist(bins=bins, label="All cycles")
ax = df_regular.duty_percentage.hist(bins=bins, alpha=1, label="Removed defrost cycles")
df_regular_non_freezer.duty_percentage.hist(bins=bins, ax=ax, alpha=1, label="Removed freezer cycles")
df_regular_non_freezer_non_hot.duty_percentage.hist(bins=bins, ax=ax, alpha=1, label="Removed hot food cycles")
df_all_above.duty_percentage.hist(bins=bins, ax=ax, alpha=1, label="Removed all fridge open cycles")

plt.legend()
plt.xlabel("Fridge compressor duty percentage")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("on.png")


# Case I
# After removing the hot food cycles, freezer cycles and defrost cycles -> 
# And when door is not opened, what is the distribution of duty percentage

df_case_1 = df_regular_non_freezer_non_hot[~df_regular_non_freezer_non_hot["Fridge opened"]]

# Case 2
# After removing the hot food cycles, freezer cycles and defrost cycles -> 
# And when door IS opened, what is the distribution of duty percentage

df_case_2 = df_regular_non_freezer_non_hot[df_regular_non_freezer_non_hot["Fridge opened"]]

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

