import pandas as pd
import matplotlib.pyplot as plt


def read_fridge_csv(csv_path):
    df = pd.read_csv(csv_path, usecols=[1, 2], skiprows=2,
                     index_col=0, names=["timestamp", "power"])
    df.index = pd.to_datetime(df.index)
    return df

paths = {
    5: "../../data/fridge/rice/5.csv",
    4: "../../data/fridge/rice/4.csv",
    3: "../../data/fridge/rice/3.csv",
    2: "../../data/fridge/rice/2.csv"
}

dfs = {}
for fridge_number, fridge_csv_path in paths.iteritems():
    print fridge_number
    dfs[fridge_number] = read_fridge_csv(csv_path=fridge_csv_path)
    dfs[fridge_number] = dfs[fridge_number]["2015-04-12":"2015-04-12 10:00"]

fig, ax = plt.subplots(nrows=4)
count = 0
for fridge_number, fridge_df in dfs.iteritems():
    print fridge_number
    fridge_df.plot(ax=ax[count])
    count += 1

plt.show()


