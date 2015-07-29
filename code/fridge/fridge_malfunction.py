import pandas as pd
import numpy as np
import sys
sys.path.append("../common")
import matplotlib.pyplot as plt
from collections import OrderedDict
import matplotlib.patches as mpatches
from common_functions import latexify, format_axes
latexify(fig_height=3)

fig, ax = plt.subplots(nrows=2, sharex=True)
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


colors = ["red","g","b","magenta","gray"]
fridge_labels =wiki_malfunction.keys()
fridge_labels.append("General Electric\n pfcs1nfzss")
def plot_malfunction(malfunction_dict, ax, title):
    for i, (fridge_name, fridge_pair_powers) in enumerate(malfunction_dict.items()):
        xs = [fridge_pair_powers[0][0], fridge_pair_powers[1][0]]
        ys = [fridge_pair_powers[0][1], fridge_pair_powers[1][1]]
        ax.scatter(xs, ys, label=fridge_name, marker='.',alpha=0.2,color=colors[i])
        print fridge_name
        print xs
        print ys

        x = (fridge_pair_powers[0][0] + fridge_pair_powers[1][0])/2
        y = (fridge_pair_powers[0][1] + fridge_pair_powers[1][1])/2
        x_start, dx  = xs[0], xs[1]-xs[0]
        y_start, dy = ys[0], ys[1]-ys[0]
        print x_start, dx, y_start, dy
        #ax.arrow(x_start, y_start, dx, dy,
        #         head_width=5, head_length=10, fc=colors[i], ec=colors[i], lw=0.7,zorder=(10-2*i))
        #  ax.annotate(percent_savings[fridge_name], xy=(x, y), xytext=(x, y),
        #            )
    #ax.set_ylabel("Transient power (W)")

    #ax.set_title(title)
    format_axes(ax)



plot_malfunction(wiki_malfunction, ax[1], "WikiEnergy data set")
ax[1].set_ylim(70,280) # outliers only
ax[0].set_ylim(1250,1430)

ax[0].scatter([111], [1303],alpha=0)
ax[1].scatter([80],[105],alpha=0)

format_axes(ax[1])
format_axes(ax[0])

ax[0].spines['top'].set_visible(False)
ax[0].spines['bottom'].set_visible(False)
ax[1].spines['top'].set_visible(False)
ax[0].xaxis.tick_top()
ax[0].tick_params(labeltop='off') # don't put tick labels at the top
#ax[1].xaxis.tick_bottom()

for i, (fridge_floor, fridge_power) in enumerate(rice_malfunction.iteritems()):
    print fridge_power
    ax[0].scatter([fridge_power[0]], [fridge_power[1]], label=fridge_floor, color=colors[i],alpha=0)

ax[1].set_xlabel("Steady state power (W)")
#ax[1].set_title("Rice Hall")
fig.text(-0.01, 0.5, 'Transient Power (W)', va='center', rotation='vertical')


#ax[0].arrow(80,1100,30,200,
#                 head_width=5, head_length=10, fc="gray", ec="gray", lw=0.7)
#ax[1].arrow(80,100,30,300,
#                 head_width=5, head_length=10, fc="gray", ec="gray", lw=0.7)

d = .015 # how big to make the diagonal lines in axes coordinates
# arguments to pass plot, just so we don't keep repeating them
kwargs = dict(transform=ax[0].transAxes, color='k', clip_on=False)
ax[0].plot((-d,+d),(-d,+d), **kwargs)      # top-left diagonal
#ax[0].plot((1-d,1+d),(-d,+d), **kwargs)    # top-right diagonal

kwargs.update(transform=ax[1].transAxes)  # switch to the bottom axes
ax[1].plot((-d,+d),(1-d,1+d), **kwargs)   # bottom-left diagonal
#ax[1].plot((1-d,1+d),(1-d,1+d), **kwargs) # bottom-right diagonal

fig.tight_layout()

patches = []
for color, label in zip(colors, fridge_labels):
    patches.append(mpatches.Patch(color=color, label=label))

fig.legend(loc='upper center', bbox_to_anchor=(0.55, 1.05),
      ncol=3,handles=patches,
       labels=fridge_labels)




plt.savefig("../../figures/fridge/identical_fridge.pdf", bbox_inches="tight")
plt.savefig("../../figures/fridge/identical_fridge.png", bbox_inches="tight")