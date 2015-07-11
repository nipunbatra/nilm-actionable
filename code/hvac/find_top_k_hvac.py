import time
import warnings

import pandas as pd
import nilmtk
from nilmtk import DataSet

warnings.filterwarnings("ignore")

import sys
import json

sys.path.append("../../code/fridge/")

if (len(sys.argv) < 2):
    ds_path = "/Users/nipunbatra/wikienergy-2013.h5"
else:
    ds_path = sys.argv[1]

d = {}
train_fraction = 0.5

print("Train fraction is ", train_fraction)

ds = DataSet(ds_path)
fridges = nilmtk.global_meter_group.select_using_appliances(type='fridge')

for b_id in ds.buildings.iterkeys():


    try:
        print("Doing for ids %d" % b_id)
        start = time.time()

        elec = ds.buildings[b_id].elec
        mains = elec.mains()

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
        ac_elec_train = train_elec[('air conditioner', 1)]
        # Finding top N appliances
        top_k_train_elec = train_elec.submeters().select_top_k(k=10)
        d[b_id] = [m.instance() for m in top_k_train_elec.meters]
    except Exception, e:
        import traceback
        traceback.print_exc()
        print e
with open('../../data/hvac/top_k_2013.json', 'w') as fp:
    json.dump(d, fp)
