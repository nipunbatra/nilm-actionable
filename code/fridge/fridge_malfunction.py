import pandas as pd
import numpy as np
import sys
sys.path.append("../common")
import matplotlib.pyplot as plt
from collections import OrderedDict

from common_functions import latexify, format_axes
latexify(fig_height=3)

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


def plot_malfunction(malfunction_dict, ax, title):
    for fridge_name, fridge_pair_powers in malfunction_dict.iteritems():
        xs = [fridge_pair_powers[0][0], fridge_pair_powers[1][0]]
        ys = [fridge_pair_powers[0][1], fridge_pair_powers[1][1]]
        ax.plot(xs, ys, label=fridge_name, marker='.')
        print fridge_name
        print xs
        print ys

        x = (fridge_pair_powers[0][0] + fridge_pair_powers[1][0])/2
        y = (fridge_pair_powers[0][1] + fridge_pair_powers[1][1])/2
        x_start, dx  = xs[1], xs[0]-xs[1]
        y_start, dy = ys[1], ys[0]-ys[1]
        print x_start, dx, y_start, dy
        #ax.arrow(xs[1], ys[1], xs[0]-xs[1], ys[0]-ys[1], head_width=0.05, head_length=0.1, fc='k', ec='k')
        #ax.annotate(percent_savings[fridge_name], xy=(x, y), xytext=(x, y),
        #            )
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