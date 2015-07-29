import numpy as np
import pandas as pd
from os.path import join
import os
from pylab import rcParams
import matplotlib.pyplot as plt
import sys
import nilmtk
from nilmtk import DataSet, TimeFrame, MeterGroup, HDFDataStore
from nilmtk.disaggregate import CombinatorialOptimisation, FHMM, Hart85
from nilmtk.utils import print_dict
from nilmtk.metrics import f1_score

import warnings
warnings.filterwarnings("ignore")

ds = DataSet("/Users/nipunbatra/Downloads/wikienergy-2.h5")
fridges = nilmtk.global_meter_group.select_using_appliances(type='fridge')

f_id = 110
b_id = 236
elec = ds.buildings[b_id].elec
mains = elec.mains()

h = Hart85()
h.train(elec.mains())

h2 = h.pair_df['T1 Active']

sys.path.append("../common")

from common_functions import latexify, format_axes

latexify(fig_height=1.2)
ax = h2[(h2<500)].hist(bins=10, color="gray",alpha=0.4)
plt.xlabel("Magnitude of rising and falling edge pairs")
plt.ylabel("Frequency")
format_axes(plt.gca())
plt.grid(False)
plt.title("")
plt.tight_layout()
plt.savefig("../../figures/fridge/hart_level.pdf", bbox_inches="tight")