import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import scipy.interpolate as inter

fig, ax = plt.subplots()

x_num = 5
xpos = np.linspace(0, 1, x_num)

plt.xlim((-0.2,1.2))
for x in xpos:
    plt.axvspan(xmin=x,xmax=x,ymin=0,ymax=1,alpha=0.5)

df = pd.read_csv("iris.csv")
numeric_cols = [col for col in df.columns if col!="Name"]
df[numeric_cols] = df[numeric_cols].div(df[numeric_cols].max())

for i, column in enumerate(df[numeric_cols]):
    plt.scatter(xpos[i]*np.ones(len(df)),df[column],alpha=0)
plt.xticks(xpos, numeric_cols)

x = xpos
mapping = {"Iris-setosa":{"color":"red", "value":0},
           "Iris-versicolor":{"color":"blue","value":0.5},
           'Iris-virginica':{"color":"green","value":1}}

for i in range(len(df)):
    y_df = df.ix[i]
    y = df.ix[i][numeric_cols].values.astype(float)
    y = np.append(y, mapping[y_df.Name]["value"])
    s1 = inter.InterpolatedUnivariateSpline (x, y,k=4)
    plt.plot(x, s1(x), color=mapping[y_df.Name]["color"],zorder=mapping[y_df.Name]["value"]*10)
plt.show()