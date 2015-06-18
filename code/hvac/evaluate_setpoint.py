import os

import numpy as np
import matplotlib.pyplot as plt

plt.style.use('ggplot')
import pandas as pd
from lmfit import minimize, Parameters


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
    #return 1
    #return mins
    MIN_MINS=0
    return (mins > MIN_MINS).astype('int')
    if mins < MIN_MINS*np.ones(len(mins)):
        return 0
    else:
        return 1

def is_used_hvac(mins):
    #return 1
    #return mins
    MIN_MINS=0
    return (mins > MIN_MINS).astype('int')
    if mins < MIN_MINS*np.ones(len(mins)):
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
        ((x[24] - v['t0']) * x[5] * is_used_hvac(x[27])) +
        ((x[24] - v['t1']) * x[6] * is_used_hvac(x[27])) +
        ((x[24] - v['t1']) * x[7] * is_used_hvac(x[27])) +
        ((x[24] - v['t1']) * x[8] * is_used_hvac(x[27])) +
        ((x[24] - v['t1']) * x[9] * is_used_hvac(x[27])) +
        ((x[24] - v['t2']) * x[10] * is_used_hvac(x[27])) +
        ((x[24] - v['t2']) * x[11] * is_used_hvac(x[27])) +
        ((x[24] - v['t2']) * x[12] * is_used_hvac(x[27])) +
        ((x[24] - v['t2']) * x[13] * is_used_hvac(x[27])) +
        ((x[24] - v['t2']) * x[14] * is_used_hvac(x[27])) +
        ((x[24] - v['t2']) * x[15] * is_used_hvac(x[27])) +
        ((x[24] - v['t2']) * x[16] * is_used_hvac(x[27])) +
        ((x[24] - v['t2']) * x[17] * is_used_hvac(x[27])) +
        ((x[24] - v['t3']) * x[18] * is_used_hvac(x[27])) +
        ((x[24] - v['t3']) * x[19] * is_used_hvac(x[27])) +
        ((x[24] - v['t3']) * x[20] * is_used_hvac(x[27])) +
        ((x[24] - v['t3']) * x[21] * is_used_hvac(x[27])) +
        ((x[24] - v['t0']) * x[22] * is_used_hvac(x[27])) +
        ((x[24] - v['t0']) * x[23] * is_used_hvac(x[27]))

    )
    model2 = v['a2'] * x[25]
    model3 = v['a3'] * x[26]

    setpoints = np.array([v['t'+str(i)] for i in range(24)])
    return np.square(model1 + model2 + model3 - data)

def fcn2min_time_fixed_binary(params, x, data):
    v = params.valuesdict()
    model1 = v['a1'] * (
        ((x[24] - v['t0']) * x[0] * is_used_hvac_binary(x[27])) +
        ((x[24] - v['t0']) * x[1] * is_used_hvac_binary(x[27])) +
        ((x[24] - v['t0']) * x[2] * is_used_hvac_binary(x[27])) +
        ((x[24] - v['t0']) * x[3] * is_used_hvac_binary(x[27])) +
        ((x[24] - v['t0']) * x[4] * is_used_hvac_binary(x[27])) +
        ((x[24] - v['t0']) * x[5] * is_used_hvac_binary(x[27])) +
        ((x[24] - v['t1']) * x[6] * is_used_hvac_binary(x[27])) +
        ((x[24] - v['t1']) * x[7] * is_used_hvac_binary(x[27])) +
        ((x[24] - v['t1']) * x[8] * is_used_hvac_binary(x[27])) +
        ((x[24] - v['t1']) * x[9] * is_used_hvac_binary(x[27])) +
        ((x[24] - v['t2']) * x[10] * is_used_hvac_binary(x[27])) +
        ((x[24] - v['t2']) * x[11] * is_used_hvac_binary(x[27])) +
        ((x[24] - v['t2']) * x[12] * is_used_hvac_binary(x[27])) +
        ((x[24] - v['t2']) * x[13] * is_used_hvac_binary(x[27])) +
        ((x[24] - v['t2']) * x[14] * is_used_hvac_binary(x[27])) +
        ((x[24] - v['t2']) * x[15] * is_used_hvac_binary(x[27])) +
        ((x[24] - v['t2']) * x[16] * is_used_hvac_binary(x[27])) +
        ((x[24] - v['t2']) * x[17] * is_used_hvac_binary(x[27])) +
        ((x[24] - v['t3']) * x[18] * is_used_hvac_binary(x[27])) +
        ((x[24] - v['t3']) * x[19] * is_used_hvac_binary(x[27])) +
        ((x[24] - v['t3']) * x[20] * is_used_hvac_binary(x[27])) +
        ((x[24] - v['t3']) * x[21] * is_used_hvac_binary(x[27])) +
        ((x[24] - v['t0']) * x[22] * is_used_hvac_binary(x[27])) +
        ((x[24] - v['t0']) * x[23] * is_used_hvac_binary(x[27]))

    )
    model2 = v['a2'] * x[25]
    model3 = v['a3'] * x[26]

    setpoints = np.array([v['t'+str(i)] for i in range(24)])
    return np.square(model1 + model2 + model3 - data)

df = pd.read_csv("../../data/total/survey_2013.csv")

cols = ['temp_summer_sleeping_hours_hours',
        'temp_summer_weekday_morning',
        'temp_summer_weekday_workday',
        'temp_summer_weekday_evening']


function_map={"binary":fcn2min_time_fixed_binary,
              "minutes":fcn2min_time_fixed}

out = {"binary":{}, "minutes":{}}


for building_string in st.keys()[::2]:
    building_num_str = building_string[1:-2]
    building_num = int(building_num_str)
    energy = st[str(building_num) + "_Y"]
    hour_usage_df = st[str(building_num) + "_X"]

    header_known = hour_usage_df.columns.tolist()
    header_unknown = ["t%d" % i for i in range(24)]
    header_unknown.append("a1")
    header_unknown.append("a2")
    header_unknown.append("a3")

    if hour_usage_df["mins_used"].sum()>200:
        print building_num

        # create a set of Parameters


        for function_name, function in function_map.iteritems():
            params = Parameters()
            for i in range(24):
                params.add('t%d' % i, value=70, min=60, max=90)
            for i in range(1, 4):
                params.add('a%d' % i, value=1)
        #   params.add('constant', value=1)


            x = hour_usage_df.T.values
            data = energy.values
            result = minimize(function, params, args=(x, data))
            final = data + result.residual

            final = data + result.residual
            setpoints = [params['t%d' % i].value for i in range(4)]
            gt_df_row = df[df.dataid == building_num][cols]

            out[function_name][building_num] = {
            'residual':np.mean(result.residual),
            'sleep': gt_df_row['temp_summer_sleeping_hours_hours'].values[0] - setpoints[0],
            'morning': gt_df_row['temp_summer_weekday_morning'].values[0] - setpoints[1],
            'work': gt_df_row['temp_summer_weekday_workday'].values[0] - setpoints[2],
            'evening': gt_df_row['temp_summer_weekday_evening'].values[0] - setpoints[3]
        }

results = {}
results["binary"] = pd.DataFrame(out["binary"]).T
results["minutes"] = pd.DataFrame(out["minutes"]).T
temp_cols = [col for col in results["binary"].columns if col!="residual"]

results["binary"][temp_cols].boxplot()
plt.savefig("temp_binary.png")

results["minutes"][temp_cols].boxplot()
plt.savefig("temp_minutes.png")

results["binary"][["residual"]].boxplot()
plt.savefig("binary_residual.png")

results["minutes"][["residual"]].boxplot()
plt.savefig("minutes_residual.png")
