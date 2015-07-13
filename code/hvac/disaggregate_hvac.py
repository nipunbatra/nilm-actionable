import os
import time
import warnings

import pandas as pd
from nilmtk import DataSet, MeterGroup
from nilmtk.disaggregate import CombinatorialOptimisation, FHMM, Hart85
from nilmtk.feature_detectors.steady_states import find_steady_states

warnings.filterwarnings("ignore")

import sys
import json

script_path = os.path.dirname(os.path.realpath(__file__))
BASH_RUN_HVAC = os.path.join(script_path, "..", "bash_runs_hvac")

if len(sys.argv) < 2:
    ds_path = "/Users/nipunbatra/wikienergy2013.h5"
else:
    ds_path = sys.argv[1]
with open(os.path.join(script_path, "..", "..", 'data/hvac/top_k_2013.json'), 'r') as fp:
    top_k_dict = json.load(fp)

num_states = int(sys.argv[2])
K = int(sys.argv[3])
train_fraction = int(sys.argv[4]) / 100.0
classifier = sys.argv[5]

print("*" * 80)
print("Arguments")

print("Number states", num_states)
print("Train fraction is ", train_fraction)
print("Top k", K)
print("Classifier", classifier)


out_file_name = "N%d_K%d_T%s_%s" % (num_states, K, sys.argv[4], classifier)

ds = DataSet(ds_path)


def find_specific_appliance(appliance_name, appliance_instance, list_of_elecs):
    for elec_name in list_of_elecs:
        appl = elec_name.appliances[0]
        if (appl.identifier.type, appl.identifier.instance) == (appliance_name, appliance_instance):
            return elec_name


out = {}
for b_id, building in ds.buildings.iteritems():
    try:
        print b_id

        out[b_id] = {}
        start = time.time()
        #cls_dict = {"Hart":Hart85()}
        cls_dict = {"CO": CombinatorialOptimisation(), "FHMM": FHMM(), "Hart": Hart85()}
        elec = building.elec
        mains = elec.mains()

        train = DataSet(ds_path)
        test = DataSet(ds_path)
        split_point = elec.train_test_split(train_fraction=train_fraction).date()
        train.set_window(end=split_point)
        #test.set_window(start=split_point)
        train_elec = train.buildings[b_id].elec
        test_elec = test.buildings[b_id].elec
        test_mains = test_elec.mains()

        # AC elec
        ac_elec_train = train_elec[('air conditioner', 1)]
        ac_elec_test = test_elec[('air conditioner', 1)]

        num_states_dict = {ac_elec_train: num_states}


        # Finding top N appliances
        top_k_train_list = top_k_dict[str(b_id)][:K]
        print("Top %d list is " % (K), top_k_train_list)
        top_k_train_elec = MeterGroup([m for m in ds.buildings[b_id].elec.meters if m.instance() in top_k_train_list])

        if not os.path.exists("%s/%s/" % (BASH_RUN_HVAC, out_file_name)):
            os.makedirs("%s/%s" % (BASH_RUN_HVAC, out_file_name))

        # Add this ac to training if this fridge is not in top-k
        if ac_elec_train not in top_k_train_elec.meters:
            top_k_train_elec.meters.append(ac_elec_train)

        try:
            clf_name = classifier
            clf = cls_dict[clf_name]

            if clf_name == "Hart":
                ac_df_train = ac_elec_train.load().next()[('power', 'active')]
                ac_power = ac_df_train[ac_df_train > 20]
                clf.train(train_elec.mains())
                d = (clf.centroids - ac_power.mean()).abs()
                ac_num = d[('power','active')].argmin()
                ac_identifier_tuple = ('unknown', ac_num)
            else:
                clf.train(top_k_train_elec, num_states_dict=num_states_dict)
                ac_instance = 1
                ac_identifier_tuple = ('air conditioner', ac_instance)

            print("-" * 80)
            print("Disaggregating")
            print("-" * 80)
            test_mains_df = test_mains.load().next()
            if clf_name == "Hart":
                [_, transients] = find_steady_states(test_mains_df, clf.cols,
                                                     clf.state_threshold, clf.noise_level)
                pred_df_ac = clf.disaggregate_chunk(test_mains_df, {}, transients)[[ac_num]]

                pred_ser_ac = pred_df_ac.squeeze()
                pred_ser_ac.name = "Hart"
                out[b_id][clf_name] = pred_ser_ac
            elif clf_name == "CO":
                pred_df = clf.disaggregate_chunk(test_mains_df)
                pred_df.columns = [clf.model[i]['training_metadata'] for i in pred_df.columns]
                pred_df_ac = pred_df[[find_specific_appliance('air conditioner',
                                                                  ac_instance,
                                                                  pred_df.columns.tolist())]]
                pred_ser_ac = pred_df_ac.squeeze()
                pred_ser_ac.name = "CO"
                out[b_id][clf_name] = pred_ser_ac
            else:
                pred_df = clf.disaggregate_chunk(test_mains_df)
                pred_df_ac = pred_df[[find_specific_appliance('air conditioner',
                                                                  ac_instance,
                                                                  pred_df.columns.tolist())]]
                pred_ser_ac = pred_df_ac.squeeze()
                pred_ser_ac.name = "FHMM"
                out[b_id][clf_name] = pred_ser_ac

            ac_df_test = ac_elec_test.load().next()[('power', 'active')]
            ac_df_test.name = "GT"
            out[b_id]["GT"] = ac_df_test
            out_df = pd.DataFrame(out[b_id])
            print("Writing for AC id: %d" % b_id)
            out_df.to_hdf("%s/%s/%d.h5" % (BASH_RUN_HVAC, out_file_name, b_id), "disag")

            end = time.time()
            time_taken = int(end - start)
            print "Id: %d took %d seconds" % (b_id, time_taken)
        except Exception, e:
            import traceback

            traceback.print_exc()
            print e
    except Exception, e:
        import traceback
        traceback.print_exc()
