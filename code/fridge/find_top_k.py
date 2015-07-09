import numpy as np
import pandas as pd
from os.path import join
import os
import time
from pylab import rcParams
import matplotlib.pyplot as plt

import nilmtk
from nilmtk import DataSet, TimeFrame, MeterGroup, HDFDataStore
from nilmtk.disaggregate import CombinatorialOptimisation, FHMM, Hart85
from nilmtk.utils import print_dict
from nilmtk.metrics import f1_score

import warnings

warnings.filterwarnings("ignore")

import sys
import shelve

sys.path.append("../../code/fridge/")

if (len(sys.argv) < 2):
    ds_path = "/Users/nipunbatra/Downloads/wikienergy-2.h5"
else:
    ds_path = sys.argv[1]

d = shelve.open("../../data/fridge/top_k.pkl")
train_fraction = 0.5

print("Train fraction is ", train_fraction)

ds = DataSet(ds_path)
fridges = nilmtk.global_meter_group.select_using_appliances(type='fridge')

fridges_id_building_id = {i: fridges.meters[i].building() for i in range(len(fridges.meters))}

fridge_id_building_id_ser = pd.Series(fridges_id_building_id)

from fridge_compressor_durations_optimised_jul_7 import compressor_powers, defrost_power

fridge_ids_to_consider = compressor_powers.keys()

building_ids_to_consider = fridge_id_building_id_ser[fridge_ids_to_consider]

for f_id, b_id in building_ids_to_consider.iteritems():
    print("Doing for ids %d and %d" % (f_id, b_id))
    start = time.time()


    elec = ds.buildings[b_id].elec
    mains = elec.mains()
    fridge_instance = fridges.meters[f_id].appliances[0].identifier.instance
    # Dividing train, test

    train = DataSet(ds_path)
    test = DataSet(ds_path)
    split_point = elec.train_test_split(train_fraction=train_fraction).date()
    train.set_window(end=split_point)
    test.set_window(start=split_point)
    train_elec = train.buildings[b_id].elec
    test_elec = test.buildings[b_id].elec
    test_mains = test_elec.mains()

    # Fridge elec
    fridge_elec_train = train_elec[('fridge', fridge_instance)]
    # Finding top N appliances
    top_k_train_elec = train_elec.submeters().select_top_k(k=10)
    d[str(f_id)] = [m.instance() for m in top_k_train_elec.meters]
d.close()
