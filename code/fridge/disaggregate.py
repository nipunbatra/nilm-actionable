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

sys.path.append("../../code/fridge/")

if (len(sys.argv) < 2):
    ds_path = "/Users/nipunbatra/Downloads/wikienergy-2.h5"
else:
    ds_path = sys.argv[1]

num_states = int(sys.argv[2])
K = int(sys.argv[3])
train_fraction = int(sys.argv[4]) / 100.0

print("Train fraction is ", train_fraction)

out_file_name = "N%d_K%d_T%d" % (num_states, K, train_fraction)

ds = DataSet(ds_path)
fridges = nilmtk.global_meter_group.select_using_appliances(type='fridge')

fridges_id_building_id = {i: fridges.meters[i].building() for i in range(len(fridges.meters))}

fridge_id_building_id_ser = pd.Series(fridges_id_building_id)

from fridge_compressor_durations_optimised_jul_7 import compressor_powers, defrost_power

fridge_ids_to_consider = compressor_powers.keys()

building_ids_to_consider = fridge_id_building_id_ser[fridge_ids_to_consider]

out = {}
for f_id, b_id in building_ids_to_consider.head(2).iteritems():
    print("Doing for ids %d and %d" %(f_id, b_id))
    start = time.time()
    out[f_id] = {}
    # Need to put it here to ensure that we have a new instance of the algorithm each time
    cls_dict = {"CO": CombinatorialOptimisation(), "FHMM": FHMM(), "Hart": Hart85()}
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
    fridge_elec_test = test_elec[('fridge', fridge_instance)]

    num_states_dict = {fridge_elec_train: num_states}


    # Finding top N appliances
    top_k_train_elec = train_elec.submeters().select_top_k(k=K)

    # Creating a folder for each classifier
    #print clf_name


    print ("../../bash_runs/%s" % (out_file_name))
    if not os.path.exists("../../bash_runs/%s" % (out_file_name)):
        os.makedirs("../../bash_runs/%s" %(out_file_name))

    for clf_name in cls_dict.keys():
        if not os.path.exists("../../bash_runs/%s/%s" %(out_file_name, clf_name)):
            os.makedirs("../../bash_runs/%s/%s" %(out_file_name, clf_name))

    # Add this fridge to training if this fridge is not in top-k
    if fridge_elec_train not in top_k_train_elec.meters:
        top_k_train_elec.meters.append(fridge_elec_train)

    try:
        for clf_name, clf in cls_dict.iteritems():
            disag_filename = '%s/%d.h5' % (clf_name, f_id)
            ds_filename_total="../../bash_runs/%s/%s" % (out_file_name, disag_filename)
            if not os.path.exists(ds_filename_total):
                # We've already learnt the model, move ahead!
                if clf_name == "Hart":
                    fridge_df_train = fridge_elec_train.load().next()[('power', 'active')]
                    fridge_power = fridge_df_train[fridge_df_train > 20]
                    clf.train(train_elec.mains())
                    d = (clf.centroids - fridge_power).abs()
                    fridge_num = d.sort(ascending=True).head(1).index.values[0]
                    fridge_identifier_tuple = ('unknown', fridge_num)
                else:
                    clf.train(top_k_train_elec, num_states_dict=num_states_dict)
                    fridge_instance = fridges.meters[f_id].appliances[0].identifier.instance
                    fridge_identifier_tuple = ('fridge', fridge_instance)

                output = HDFDataStore(ds_filename_total, 'w')
                clf.disaggregate(test_mains, output)
                output.close()

                # Now, need to grab the DF
                ds_pred = DataSet(ds_filename_total)
                out[f_id][clf_name] = ds_pred.buildings[b_id].elec[fridge_identifier_tuple].load().next()[
                    ('power', 'active')]
                fridge_df_test = fridge_elec_test.load().next()[('power', 'active')]
                out[f_id]["GT"] = fridge_df_test
                out_df = pd.DataFrame(out[f_id])
                if not os.path.exists("../../bash_runs/%s/output/" %(out_file_name)):
                    os.makedirs("../../bash_runs/%s/output/" %(out_file_name))
                out_df.to_hdf("../../bash_runs/%s/output/%d.h5" % (out_file_name,f_id), "disag")

            else:
                print("Skipping")
            end = time.time()
            time_taken = int(end - start)
            print "Id: %d took %d seconds" % (f_id, time_taken)
    except Exception, e:
        import traceback
        traceback.print_exc()
        print e
