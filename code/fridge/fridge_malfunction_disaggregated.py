import pandas as pd
import numpy as np
import sys
sys.path.append("../common")
import matplotlib.pyplot as plt
from collections import OrderedDict

from common_functions import latexify, format_axes
latexify()

def make_pred(algo, num):
    if algo=="CO":
        return co(num)
    elif algo=="Hart":
        return hart(num)
    else:
        return fhmm(num)


def co(num):
    return 0.9*num+4.31

def fhmm(num):
    return 0.9*num+7.66

def hart(num):
    return 0.6*num+56.41

fig, ax = plt.subplots(nrows=2)
wiki_malfunction = {
    "Frigidaire\n frt18nrjw1": [[139, 160], [188, 255]],
    "Samsung \n rf266abrs": [[107, 141], [130, 162]],
    "LG \n lfx25960st": [[170, 219], [183, 248]],
    "Samsung \n rfg297aars": [[126, 164], [136, 175]]
}

percent_savings = {
    "Frigidaire\n frt18nrjw1": 26,
    "Samsung \n rf266abrs": 18,
    "LG \n lfx25960st": 7,
    "Samsung \n rfg297aars": 7
}

rice_malfunction = OrderedDict({
    "2nd floor": [111, 1303],
    "3rd floor": [81, 109],
    "4th floor": [81, 116],
    "5th floor": [80, 109]
})

gt = []
preds = {"FHMM":[], "CO":[],"Hart":[], "GT":[]}

names = []
for fridge_name, fridge_pair_powers in wiki_malfunction.iteritems():
    xs = [fridge_pair_powers[0][0], fridge_pair_powers[1][0]]
    names.append(fridge_name)
    gt.append(xs)
    #preds["GT"].append(xs[0])
    #preds["GT"].append(xs[1])
    for algo in preds.keys():
        preds[algo].append(make_pred(algo, xs[0]))
        preds[algo].append(make_pred(algo, xs[1]))

df = pd.DataFrame(preds)
df["GT"] = np.array(gt).flatten()

df_copy = df.copy()
a =[]
for i in range(4):
    a.append((df.ix[2*i+1]-df.ix[2*i]).div(df.ix[2*i+1]))

savings = pd.DataFrame(a).mul(100)
savings.index=names
savings.plot(kind="bar", rot=0)
plt.xlabel("Fridge pairs")
plt.ylabel(r"$\%$ energy savings")
plt.tight_layout()
plt.savefig("../../figures/fridge/disag_malfunction.pdf", bbox_inches="tight")
plt.savefig("../../figures/fridge/disag_malfunction.png", bbox_inches="tight")


print "HERE"


"""

        ys = [fridge_pair_powers[0][1], fridge_pair_powers[1][1]]
        ax.plot(xs, ys, label=fridge_name, linestyle='-', marker='o')

        x = (fridge_pair_powers[0][0] + fridge_pair_powers[1][0])/2
        y = (fridge_pair_powers[0][1] + fridge_pair_powers[1][1])/2
        print x,y
        ax.annotate(percent_savings[fridge_name], xy=(x, y), xytext=(x, y),
                    )
    ax.set_ylabel("Transient power (W)")

    ax.set_title(title)
    format_axes(ax)



plot_malfunction(wiki_malfunction, ax[0], "WikiEnergy data set")

colors = ["b", "r", "g", "k"]
for i, (fridge_floor, fridge_power) in enumerate(rice_malfunction.iteritems()):
    print fridge_power
    ax[1].scatter([fridge_power[0]], [fridge_power[1]], label=fridge_floor, color=colors[i])

ax[1].set_xlabel("Steady state power (W)")
ax[1].set_title("Rice Hall")
ax[1].set_ylabel("Transient power (W)")
format_axes(ax[1])

plt.tight_layout()

for a in ax:
    box = a.get_position()
    a.set_position([box.x0, box.y0, box.width * 0.66, box.height])

    # Put a legend to the right of the current axis
    a.legend(loc='center left', bbox_to_anchor=(1, 0.5))

plt.savefig("../../figures/fridge/identical_fridge.pdf", bbox_inches="tight")
plt.savefig("../../figures/fridge/identical_fridge.png", bbox_inches="tight")
"""