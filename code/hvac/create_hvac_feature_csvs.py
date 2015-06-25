import os

import pandas as pd
import numpy as np

START, STOP = '2013-07-01', '2013-07-31'


def num_from_key(key):
    return int(key[1:])


def key_from_num(num):
    return "/" + str(num)


store = pd.HDFStore("/Users/nipunbatra/Downloads/wiki-temp.h5")

a = store.keys()
ids = map(num_from_key, a)

df = pd.read_csv("../../data/total/survey_2013.csv")
survey_homes = df.dataid.values

data_homes = np.array(ids)

ids_common = np.intersect1d(survey_homes, data_homes)

cols = ['dataid', 'programmable_thermostat_currently_programmed',
        'temp_summer_weekday_workday', 'temp_summer_weekday_morning',
        'temp_summer_weekday_evening', 'temp_summer_sleeping_hours_hours']

id_to_use = []
for id_home in ids_common[:]:
    try:

        d = store[key_from_num(id_home)]['air1'][START:STOP]
        if len(d) > 0:
            power = d
            mins_used = (power > 500).astype('int').resample("1H", how="sum")
            if mins_used.sum() > 200:
                id_to_use.append(id_home)
    except Exception, e:
        print id_home
        print e

df_res = df[df.dataid.isin(id_to_use)]
print len(id_to_use)

df_res = df_res[cols]

from lmfit import minimize, Parameters


def assign_class_from_count(x):
    if x < 2:
        return "Bad"
    if x >= 2 and x < 3:
        return "Average"
    else:
        return "Good"


def assign_hvac_score(sleep_gt, morning_gt, work_gt, evening_gt):
    count = 0
    if work_gt >= 85:
        count += 1
    elif work_gt >= 78:
        count += (work_gt - 78) / 7.0
    if morning_gt >= 78:
        count += 1
    if evening_gt >= 78:
        count += 1
    if sleep_gt >= 82:
        count += 1
    return count

def assign_hvac_class(sleep_gt, morning_gt, work_gt, evening_gt):
    count = assign_hvac_score(sleep_gt, morning_gt, work_gt, evening_gt)
    return assign_class_from_count(count)


def get_hourly_aggregate(df, how='mean'):
    df_c = df.copy()
    df_c["hour"] = df_c.index.hour
    return df_c.groupby("hour").mean()

# Weather data store
WEATHER_DATA_STORE = "../../data/hvac/weather_2013.h5"

# Weather and HVAC data store
WEATHER_HVAC_STORE = "../../data/hvac/weather_hvac_2013.h5"
assert os.path.isfile(WEATHER_HVAC_STORE), "File does not exist"

st = pd.HDFStore(WEATHER_HVAC_STORE)


def is_used_hvac_binary(mins):
    # return 1
    # return mins
    MIN_MINS = 0
    return (mins > MIN_MINS).astype('int')
    if mins < MIN_MINS * np.ones(len(mins)):
        return 0
    else:
        return 1


def is_used_hvac(mins):
    # return 1
    # return mins
    MIN_MINS = 0
    return (mins > MIN_MINS).astype('int')
    if mins < MIN_MINS * np.ones(len(mins)):
        return 0
    else:
        return 1


def fcn2min_time_fixed(params, x, data):
    v = params.valuesdict()
    model1 = v['a1'] * (
    ((x[24] - v['t0']) * x[0] * is_used_hvac(x[27])) +
    ((x[24] - v['t0']) * x[1] * is_used_hvac(x[27])) +
    ((x[24] - v['t0']) * x[2] * is_used_hvac(x[27])) +
    ((x[24] - v['t0']) * x[3] * is_used_hvac(x[27])) +
    ((x[24] - v['t0']) * x[4] * is_used_hvac(x[27])) +
    ((x[24] - v['t0']) * x[5] * is_used_hvac(x[27]))) + \
    v['a2']*(
    ((x[24] - v['t1']) * x[6] * is_used_hvac(x[27])) +
    ((x[24] - v['t1']) * x[7] * is_used_hvac(x[27])) +
    ((x[24] - v['t1']) * x[8] * is_used_hvac(x[27])) +
    ((x[24] - v['t1']) * x[9] * is_used_hvac(x[27])) +
    ((x[24] - v['t2']) * x[10] * is_used_hvac(x[27])) +
    ((x[24] - v['t2']) * x[11] * is_used_hvac(x[27]))) +\
    v['a3']*(
    ((x[24] - v['t2']) * x[12] * is_used_hvac(x[27])) +
    ((x[24] - v['t2']) * x[13] * is_used_hvac(x[27])) +
    ((x[24] - v['t2']) * x[14] * is_used_hvac(x[27])) +
    ((x[24] - v['t2']) * x[15] * is_used_hvac(x[27])) +
    ((x[24] - v['t2']) * x[16] * is_used_hvac(x[27])) +
    ((x[24] - v['t2']) * x[17] * is_used_hvac(x[27]))) +\
    v['a4'] * (
    ((x[24] - v['t3']) * x[18] * is_used_hvac(x[27])) +
    ((x[24] - v['t3']) * x[19] * is_used_hvac(x[27])) +
    ((x[24] - v['t3']) * x[20] * is_used_hvac(x[27])) +
    ((x[24] - v['t3']) * x[21] * is_used_hvac(x[27])) +
    ((x[24] - v['t0']) * x[22] * is_used_hvac(x[27])) +
    ((x[24] - v['t0']) * x[23] * is_used_hvac(x[27])))

    model2 = v['a5'] * x[25]
    model3 = v['a6'] * x[26]


    return np.square(model1 + model2 + model3 - data)

def fcn2min_time_fixed_binary(params, x, data):
        v = params.valuesdict()
        model1 = v['a1'] * (
        ((x[24] - v['t0']) * x[0] * is_used_hvac(x[27])) +
        ((x[24] - v['t0']) * x[1] * is_used_hvac(x[27])) +
        ((x[24] - v['t0']) * x[2] * is_used_hvac(x[27])) +
        ((x[24] - v['t0']) * x[3] * is_used_hvac(x[27])) +
        ((x[24] - v['t0']) * x[4] * is_used_hvac(x[27])) +
        ((x[24] - v['t0']) * x[5] * is_used_hvac(x[27]))) + \
        v['a2']*(
        ((x[24] - v['t1']) * x[6] * is_used_hvac(x[27])) +
        ((x[24] - v['t1']) * x[7] * is_used_hvac(x[27])) +
        ((x[24] - v['t1']) * x[8] * is_used_hvac(x[27])) +
        ((x[24] - v['t1']) * x[9] * is_used_hvac(x[27])) +
        ((x[24] - v['t2']) * x[10] * is_used_hvac(x[27])) +
        ((x[24] - v['t2']) * x[11] * is_used_hvac(x[27]))) +\
        v['a3']*(
        ((x[24] - v['t2']) * x[12] * is_used_hvac(x[27])) +
        ((x[24] - v['t2']) * x[13] * is_used_hvac(x[27])) +
        ((x[24] - v['t2']) * x[14] * is_used_hvac(x[27])) +
        ((x[24] - v['t2']) * x[15] * is_used_hvac(x[27])) +
        ((x[24] - v['t2']) * x[16] * is_used_hvac(x[27])) +
        ((x[24] - v['t2']) * x[17] * is_used_hvac(x[27]))) +\
        v['a4'] * (
        ((x[24] - v['t3']) * x[18] * is_used_hvac(x[27])) +
        ((x[24] - v['t3']) * x[19] * is_used_hvac(x[27])) +
        ((x[24] - v['t3']) * x[20] * is_used_hvac(x[27])) +
        ((x[24] - v['t3']) * x[21] * is_used_hvac(x[27])) +
        ((x[24] - v['t0']) * x[22] * is_used_hvac(x[27])) +
        ((x[24] - v['t0']) * x[23] * is_used_hvac(x[27])))

        model2 = v['a5'] * x[25]
        model3 = v['a6'] * x[26]


        return np.square(model1 + model2 + model3 - data)


function_map = {"binary": fcn2min_time_fixed_binary,
                "minutes": fcn2min_time_fixed}

out = {"binary": {}, "minutes": {}}

for building_string in st.keys()[::2][:]:
    building_num_str = building_string[1:-2]
    building_num = int(building_num_str)
    energy = st[str(building_num) + "_Y"]
    hour_usage_df = st[str(building_num) + "_X"]

    header_known = hour_usage_df.columns.tolist()
    header_unknown = ["t%d" % i for i in range(24)]
    header_unknown.append("a1")
    header_unknown.append("a2")
    header_unknown.append("a3")

    if hour_usage_df["mins_used"].sum() > 200:
        print building_num

        # create a set of Parameters
        for function_name, function in function_map.iteritems():
            params = Parameters()
            for i in range(4):
                params.add('t%d' % i, value=70, min=60, max=90)
            for i in range(1, 7):
                params.add('a%d' % i, value=1)
                #   params.add('constant', value=1)

            x = hour_usage_df.T.values
            data = energy.values
            result = minimize(function, params, args=(x, data))
            final = data + result.residual

            final = data + result.residual
            setpoints = [params['t%d' % i].value for i in range(4)]
            gt_df_row = df[df.dataid == building_num][cols]
            power_df = store['/%d' % building_num]['air1'][START:STOP]
            mins = (power_df > 500).sum()

            sleep_gt = gt_df_row['temp_summer_sleeping_hours_hours'].values[0]
            morning_gt = gt_df_row['temp_summer_weekday_morning'].values[0]
            work_gt = gt_df_row['temp_summer_weekday_workday'].values[0]
            evening_gt = gt_df_row['temp_summer_weekday_evening'].values[0]

            out[function_name][building_num] = {
                'a1': params['a1'].value,
                'a2': params['a2'].value,
                'a3': params['a3'].value,
                'a4':params['a4'].value,
                'a5':params['a5'].value,
                'a6':params['a6'].value,
                'sleep_pred': setpoints[0],
                'morning_pred': setpoints[1],
                'work_pred': setpoints[2],
                'evening_pred': setpoints[3],
                'sleep_gt': sleep_gt,
                'morning_gt': morning_gt,
                'work_gt': work_gt,
                'evening_gt': evening_gt,
                'overall_mins': mins,
                'overall_energy': power_df.sum(),
                'morning_energy': power_df.between_time("05:00", "10:00").sum(),
                'work_energy': power_df.between_time("10:01", "17:00").sum(),
                'evening_energy': power_df.between_time("17:01", "22:00").sum(),
                'sleep_energy': power_df.between_time("22:01", "05:00").sum(),
                'morning_mins': (power_df.between_time("05:00", "10:00") > 500).sum(),
                'work_mins': (power_df.between_time("10:01", "17:00") > 500).sum(),
                'evening_mins': (power_df.between_time("17:01", "22:00") > 500).sum(),
                'sleep_mins': (power_df.between_time("22:01", "05:00") > 500).sum(),
                'hvac_class': assign_hvac_class(sleep_gt, morning_gt, work_gt, evening_gt),
                'rating': assign_hvac_score(sleep_gt, morning_gt, work_gt, evening_gt)
            }

results = {}
results["binary"] = pd.DataFrame(out["binary"]).T
results["minutes"] = pd.DataFrame(out["minutes"]).T

results["binary"].to_csv("../../data/hvac/binary_a3_score.csv", index_label="dataid")
results["minutes"].to_csv("../../data/hvac/minutes_a3_score.csv", index_label="dataid")
