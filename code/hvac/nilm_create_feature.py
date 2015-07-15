import os

import numpy as np
import pandas as pd
import nilmtk
import glob

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



to_consider = ["N2_K3_T50_CO","N3_K3_T50_CO",
               "N2_K3_T50_FHMM","N3_K3_T50_FHMM",
               "N2_K3_T50_CO","N2_K3_T50_Hart" ]

script_path = os.path.dirname(os.path.realpath(__file__))
weather_params = ["temperature", "humidity", "windSpeed"]

data_folder = os.path.join(script_path,"..","bash_runs_hvac")
START, STOP = '2013-07-01', '2013-07-31'

def num_from_key(key):
    return int(key[1:])

def key_from_num(num):
    return "/"+str(num)

def get_hourly_aggregate(df, how='mean'):
    df_c = df.copy()
    df_c["hour"] = df_c.index.hour
    return df_c.groupby("hour").mean()



def get_hourly_data(df, values):
    df_c = df.copy()
    df_c["day"] = df_c.index.dayofyear
    df_c["hour"] = df_c.index.hour
    return pd.pivot_table(
        df_c, index=["hour"], columns=["day"], values=values)

# Weather data store
WEATHER_DATA_STORE = os.path.join(script_path, "../../data/hvac/weather_2013.h5")
weather_data_df = pd.HDFStore(WEATHER_DATA_STORE)["/weather"]
df = pd.read_csv(os.path.join(script_path, "../../data/total/survey_2013.csv"))
cols = ['programmable_thermostat_currently_programmed',
        'temp_summer_weekday_workday', 'temp_summer_weekday_morning',
        'temp_summer_weekday_evening', 'temp_summer_sleeping_hours_hours']
from copy import deepcopy

cols_plus_data_id = deepcopy(cols)
cols_plus_data_id.insert(0, "dataid")
df = df[cols_plus_data_id].dropna()
survey_homes = df.dataid.values

ds = nilmtk.DataSet(os.path.expanduser("~/wikienergy-2013.h5"))
nilmtk_to_dataid = {{num: building.metadata["original_name"]
                     for num, building in ds.buildings.iteritems()}}
dataid_to_nilmtk={v:k for k, v in nilmtk_to_dataid.iteritems()}

function_map = {"binary": fcn2min_time_fixed_binary,
                "minutes": fcn2min_time_fixed}


for folder in to_consider:
    output = {"binary": {}, "minutes": {}}

    algo = folder.split("_")[-1]
    full_path = os.path.join(data_folder, folder)
    homes = glob.glob(full_path+"/*.h5")
    home_numbers = [int(h.split("/")[-1].split(".")[0]) for h in homes]
    home_numbers_dataid = [nilmtk_to_dataid[x] for x in home_numbers]
    data_homes = np.array(home_numbers_dataid)
    ids_common = np.intersect1d(survey_homes, data_homes)
    for id_home_data_id in ids_common:
        nilmtk_id = dataid_to_nilmtk[id_home_data_id]
        home_name = os.path.join(full_path,"%d.h5" %nilmtk_id)
        with pd.HDFStore(home_name) as st:
            d = st["/disag"][START:STOP][algo]
            power = d
            mins_used = (power > 500).astype('int').resample("1H", how="sum")
            energy = power.resample("1H", how="sum")
            # Energy in kWh
            energy = energy.div(60 * 1000)
            energy.name = "energy"
            mins_used.name = "mins"
            index_intersection = weather_data_df.index.intersection(energy.index)
            weather_data_df_restricted = weather_data_df.ix[index_intersection]
            energy_restricted = energy.ix[index_intersection]
            mins_used_restricted = mins_used.ix[index_intersection]

            out = index_intersection.hour
            out_reshaped = np.zeros((len(out), 24))

            for i, hour in enumerate(out):
                out_reshaped[i, hour] = 1

            hour_cols = ["h" + str(i) for i in range(24)]
            hour_usage_df = pd.DataFrame(out_reshaped, index=index_intersection)
            hour_usage_df.columns = hour_cols

            for param in weather_params:
                hour_usage_df[param] = weather_data_df_restricted[param]

            hour_usage_df["mins_used"] = mins_used_restricted.values

            if len(energy_restricted) > 10:
                energy = energy_restricted
                hour_usage_df=hour_usage_df
                header_known = hour_usage_df.columns.tolist()
                header_unknown = ["t%d" % i for i in range(24)]
                header_unknown.append("a1")
                header_unknown.append("a2")
                header_unknown.append("a3")

                if hour_usage_df["mins_used"].sum() > 200:

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
                        gt_df_row = df[df.dataid == id_home_data_id][cols]

                        mins = (power > 500).sum()

                        sleep_gt = gt_df_row['temp_summer_sleeping_hours_hours'].values[0]
                        morning_gt = gt_df_row['temp_summer_weekday_morning'].values[0]
                        work_gt = gt_df_row['temp_summer_weekday_workday'].values[0]
                        evening_gt = gt_df_row['temp_summer_weekday_evening'].values[0]

                        output[function_name][id_home_data_id] = {
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
                            'overall_energy': power.sum(),
                            'morning_energy': power.between_time("05:00", "10:00").sum(),
                            'work_energy': power.between_time("10:01", "17:00").sum(),
                            'evening_energy': power.between_time("17:01", "22:00").sum(),
                            'sleep_energy': power.between_time("22:01", "05:00").sum(),
                            'morning_mins': (power.between_time("05:00", "10:00") > 500).sum(),
                            'work_mins': (power.between_time("10:01", "17:00") > 500).sum(),
                            'evening_mins': (power.between_time("17:01", "22:00") > 500).sum(),
                            'sleep_mins': (power.between_time("22:01", "05:00") > 500).sum(),
                            'hvac_class': assign_hvac_class(sleep_gt, morning_gt, work_gt, evening_gt),
                            'rating': assign_hvac_score(sleep_gt, morning_gt, work_gt, evening_gt)
                        }
    results = {}
    results["binary"] = pd.DataFrame(output["binary"]).T
    results["minutes"] = pd.DataFrame(output["minutes"]).T





















