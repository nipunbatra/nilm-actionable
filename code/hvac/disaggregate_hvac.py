import os
import time
import warnings

import pandas as pd
import nilmtk
from nilmtk import DataSet, MeterGroup, HDFDataStore
from nilmtk.disaggregate import CombinatorialOptimisation, FHMM, Hart85
from nilmtk.feature_detectors.steady_states import find_steady_states_transients, find_steady_states


warnings.filterwarnings("ignore")

import sys
import json


if (len(sys.argv) < 2):
    ds_path = "/Users/nipunbatra/wikienergy2013.h5"
else:
    ds_path = sys.argv[1]
with open('../../../data/hvac/top_k_2013.json', 'r') as fp:
    top_k_dict = json.load(fp)

num_states = int(sys.argv[2])
K = int(sys.argv[3])
train_fraction = int(sys.argv[4]) / 100.0

print("*"*80)
print("Arguments")

print("Number states", num_states)
print("Train fraction is ", train_fraction)
print("Top k", K)

out_file_name = "N%d_K%d_T%s" % (num_states, K, sys.argv[4])

ds = DataSet(ds_path)

def find_specific_appliance(appliance_name, appliance_instance, list_of_elecs):
    for elec_name in list_of_elecs:
        appl = elec_name.appliances[0]
        if (appl.identifier.type, appl.identifier.instance) == (appliance_name, appliance_instance):
            return elec_name



b_id = 2
cls_dict = {"CO": CombinatorialOptimisation(), "FHMM": FHMM(), "Hart": Hart85()}
elec = ds.buildings[b_id].elec
mains = elec.mains()

train = DataSet(ds_path)
test = DataSet(ds_path)
split_point = elec.train_test_split(train_fraction=train_fraction).date()
train.set_window(end=split_point)
test.set_window(start=split_point)
train_elec = train.buildings[b_id].elec
test_elec = test.buildings[b_id].elec
test_mains = test_elec.mains()

    # AC elec
ac_elec_train = train_elec[('fridge', 1)]
facelec_test = test_elec[('fridge', 1)]

num_states_dict = {ac_elec_train: num_states}


# Finding top N appliances
top_k_train_list = top_k_dict[str(f_id)][:K]
print("Top %d list is " %(K), top_k_train_list)
top_k_train_elec = MeterGroup([m for m in ds.buildings[b_id].elec.meters if m.instance() in top_k_train_list])

#print ("../../bash_runs/%s" % (out_file_name))
#if not os.path.exists("../../bash_runs/%s" % (out_file_name)):
#    os.makedirs("../../bash_runs/%s" % (out_file_name))


# Add this fridge to training if this fridge is not in top-k
if ac_elec_train not in top_k_train_elec.meters:
    top_k_train_elec.meters.append(ac_elec_train)

    try:
        for clf_name, clf in cls_dict.iteritems():
            print("-"*80)
            print("Training on %s" %clf_name)
            disag_filename = '%s/%d.h5' % (clf_name, f_id)
            ds_filename_total = "../../bash_runs/%s/%s" % (out_file_name, disag_filename)
            if not os.path.exists(ds_filename_total):
                # We've already learnt the model, move ahead!
                if clf_name == "Hart":
                    fridge_df_train = ac_elec_train.load().next()[('power', 'active')]
                    fridge_power = fridge_df_train[fridge_df_train > 20]
                    clf.train(train_elec.mains())
                    d = (clf.centroids - fridge_power).abs()
                    fridge_num = d.sort(ascending=True).head(1).index.values[0]
                    fridge_identifier_tuple = ('unknown', fridge_num)
                else:
                    clf.train(top_k_train_elec, num_states_dict=num_states_dict)
                    fridge_instance = fridges.meters[f_id].appliances[0].identifier.instance
                    fridge_identifier_tuple = ('fridge', fridge_instance)


                print("-"*80)
                print("Disaggregating")
                print("-"*80)
                test_mains_df = test_mains.load().next()
                if clf_name=="Hart":
                    [_, transients] = find_steady_states(test_mains_df, clf.cols,
                                                         clf.state_threshold, clf.noise_level)
                    pred_df_fridge = clf.disaggregate_chunk(test_mains_df,
                                                               {}, transients)[[fridge_num]]

                    pred_ser_fridge = pred_df_fridge.squeeze()
                    pred_ser_fridge.name="Hart"
                    out[f_id][clf_name]=pred_ser_fridge
                elif clf_name=="CO":
                    pred_df = clf.disaggregate_chunk(test_mains_df)
                    pred_df.columns = [clf.model[i]['training_metadata'] for i in pred_df.columns]
                    pred_df_fridge = pred_df[[find_specific_appliance('fridge',
                                                                         fridge_instance,
                                                                         pred_df.columns.tolist())]]
                    pred_ser_fridge = pred_df_fridge.squeeze()
                    pred_ser_fridge.name="CO"
                    out[f_id][clf_name]=pred_ser_fridge
                else:
                    pred_df = clf.disaggregate_chunk(test_mains_df)
                    pred_df_fridge = pred_df[[find_specific_appliance('fridge',
                                                                         fridge_instance,
                                                                         pred_df.columns.tolist())]]
                    pred_ser_fridge = pred_df_fridge.squeeze()
                    pred_ser_fridge.name="FHMM"
                    out[f_id][clf_name]=pred_ser_fridge

        fridge_df_test = fridge_elec_test.load().next()[('power', 'active')]
        fridge_df_test.name="GT"
        out[f_id]["GT"] = fridge_df_test
        out_df = pd.DataFrame(out[f_id])
        print("Writing for fridge id: %d" %f_id)
        out_df.to_hdf("../../bash_runs/%s/%d.h5" % (out_file_name, f_id), "disag")

        end = time.time()
        time_taken = int(end - start)
        print "Id: %d took %d seconds" % (f_id, time_taken)
    except Exception, e:
        import traceback

        traceback.print_exc()
        print e
