import os

import numpy as np
import matplotlib.pyplot as plt

import pandas as pd
from lmfit import minimize, Parameters, report_fit

import sys
sys.path.append("../common")

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




building_num = st.keys()[14][1:-2]
energy = st[str(building_num) + "_Y"]
hour_usage_df = st[str(building_num) + "_X"]

header_known = hour_usage_df.columns.tolist()
header_unknown = ["t%d" % i for i in range(24)]
header_unknown.append("a1")
header_unknown.append("a2")
header_unknown.append("a3")


MIN_MINS = 0

def is_used_hvac(mins):
    #return 1
    return mins
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
        ((x[24] - v['t0']) * x[23] * is_used_hvac(x[27]))

    )
    model2 = v['a5'] * x[25]
    model3 = v['a6'] * x[26]

    setpoints = np.array([v['t'+str(i)] for i in range(24)])
    return np.square(model1 + model2 + model3 - data)

def fcn2min_time(params, x, data):
    v = params.valuesdict()
    model1 = v['a1'] * (
        ((x[24] - v['t0']) * x[0] * is_used_hvac(x[27])) +
        ((x[24] - v['t1']) * x[1] * is_used_hvac(x[27])) +
        ((x[24] - v['t2']) * x[2] * is_used_hvac(x[27])) +
        ((x[24] - v['t3']) * x[3] * is_used_hvac(x[27])) +
        ((x[24] - v['t4']) * x[4] * is_used_hvac(x[27])) +
        ((x[24] - v['t5']) * x[5] * is_used_hvac(x[27])) +
        ((x[24] - v['t6']) * x[6] * is_used_hvac(x[27])) +
        ((x[24] - v['t7']) * x[7] * is_used_hvac(x[27])) +
        ((x[24] - v['t8']) * x[8] * is_used_hvac(x[27])) +
        ((x[24] - v['t9']) * x[9] * is_used_hvac(x[27])) +
        ((x[24] - v['t10']) * x[10] * is_used_hvac(x[27])) +
        ((x[24] - v['t11']) * x[11] * is_used_hvac(x[27])) +
        ((x[24] - v['t12']) * x[12] * is_used_hvac(x[27])) +
        ((x[24] - v['t13']) * x[13] * is_used_hvac(x[27])) +
        ((x[24] - v['t14']) * x[14] * is_used_hvac(x[27])) +
        ((x[24] - v['t15']) * x[15] * is_used_hvac(x[27])) +
        ((x[24] - v['t16']) * x[16] * is_used_hvac(x[27])) +
        ((x[24] - v['t17']) * x[17] * is_used_hvac(x[27])) +
        ((x[24] - v['t18']) * x[18] * is_used_hvac(x[27])) +
        ((x[24] - v['t19']) * x[19] * is_used_hvac(x[27])) +
        ((x[24] - v['t20']) * x[20] * is_used_hvac(x[27])) +
        ((x[24] - v['t21']) * x[21] * is_used_hvac(x[27])) +
        ((x[24] - v['t22']) * x[22] * is_used_hvac(x[27])) +
        ((x[24] - v['t23']) * x[23] * is_used_hvac(x[27]))

    )
    model2 = v['a2'] * x[25]
    model3 = v['a3'] * x[26]
    setpoints = np.array([v['t'+str(i)] for i in range(24)])
    return np.square(model1 + model2 + model3 - data) + \
           0.005 * np.std(setpoints[8:15]) + \
           0.0 * np.std(setpoints[16:21])

def fcn2min_penalty(params, x, data):
    v = params.valuesdict()
    model1 = v['a1'] * (
        ((x[24] - v['t0']) * x[0]) +
        ((x[24] - v['t1']) * x[1]) +
        ((x[24] - v['t2']) * x[2]) +
        ((x[24] - v['t3']) * x[3]) +
        ((x[24] - v['t4']) * x[4]) +
        ((x[24] - v['t5']) * x[5]) +
        ((x[24] - v['t6']) * x[6]) +
        ((x[24] - v['t7']) * x[7]) +
        ((x[24] - v['t8']) * x[8]) +
        ((x[24] - v['t9']) * x[9]) +
        ((x[24] - v['t10']) * x[10]) +
        ((x[24] - v['t11']) * x[11]) +
        ((x[24] - v['t12']) * x[12]) +
        ((x[24] - v['t13']) * x[13]) +
        ((x[24] - v['t14']) * x[14]) +
        ((x[24] - v['t15']) * x[15]) +
        ((x[24] - v['t16']) * x[16]) +
        ((x[24] - v['t17']) * x[17]) +
        ((x[24] - v['t18']) * x[18]) +
        ((x[24] - v['t19']) * x[19]) +
        ((x[24] - v['t20']) * x[20]) +
        ((x[24] - v['t21']) * x[21]) +
        ((x[24] - v['t22']) * x[22]) +
        ((x[24] - v['t23']) * x[23])

    )
    model2 = v['a2'] * x[25]
    model3 = v['a3'] * x[26]
    setpoints = np.array([v['t'+str(i)] for i in range(24)])
    return np.square(model1 + model2 + model3 - data) + 0.0*np.std(setpoints)


def fcn2min(params, x, data):
    v = params.valuesdict()
    model1 = v['a1'] * (
        ((x[24] - v['t0']) * x[0]) +
        ((x[24] - v['t1']) * x[1]) +
        ((x[24] - v['t2']) * x[2]) +
        ((x[24] - v['t3']) * x[3]) +
        ((x[24] - v['t4']) * x[4]) +
        ((x[24] - v['t5']) * x[5]) +
        ((x[24] - v['t6']) * x[6]) +
        ((x[24] - v['t7']) * x[7]) +
        ((x[24] - v['t8']) * x[8]) +
        ((x[24] - v['t9']) * x[9]) +
        ((x[24] - v['t10']) * x[10]) +
        ((x[24] - v['t11']) * x[11]) +
        ((x[24] - v['t12']) * x[12]) +
        ((x[24] - v['t13']) * x[13]) +
        ((x[24] - v['t14']) * x[14]) +
        ((x[24] - v['t15']) * x[15]) +
        ((x[24] - v['t16']) * x[16]) +
        ((x[24] - v['t17']) * x[17]) +
        ((x[24] - v['t18']) * x[18]) +
        ((x[24] - v['t19']) * x[19]) +
        ((x[24] - v['t20']) * x[20]) +
        ((x[24] - v['t21']) * x[21]) +
        ((x[24] - v['t22']) * x[22]) +
        ((x[24] - v['t23']) * x[23])

    )
    model2 = v['a2'] * x[25]
    model3 = v['a3'] * x[26]
    setpoints = np.array([v['t'+str(i)] for i in range(24)])
    return np.square(model1 + model2 + model3 - data) + 0.0*np.std(setpoints)


# create a set of Parameters
params = Parameters()
for i in range(24):
    params.add('t%d' % i, value=70, min=60, max=90)
for i in range(1, 7):
    params.add('a%d' % i, value=1)
#params.add('constant', value=1)

x = hour_usage_df.T.values
data = energy.values
result = minimize(fcn2min_time_fixed, params, args=(x, data))
final = data + result.residual

final = data + result.residual

# write error report
report_fit(params)

SAVE = False

setpoints = [params['t%d' % i].value for i in range(24)]
energy_df = pd.DataFrame({"energy": energy})
energy_hourly_mean_df = get_hourly_aggregate(energy_df)
temp_hourly_mean_df = get_hourly_aggregate(hour_usage_df[["temperature"]])

from common_functions import latexify, format_axes
latexify(columns=1, fig_height=3.0)
fig, ax = plt.subplots(nrows=2)
ax[0].scatter(data, final, color="gray",alpha=0.4, s=2)
ax[0].set_xlabel("Actual energy consumption(kWh)\n(a)")
ax[0].set_ylabel("Predicted energy\n consumption(kWh)")

ax[1].plot(data[:24], label='Actual')
ax[1].plot(final[:24], label='Predicted')
#plt.fill_between(range(len(data[:24])), data[:24], 0, color='g', alpha=1, label='actual')
#plt.fill_between(range(len(final[:24])), final[:24], 0, color='r', alpha=0.5, label='predicted')
ax[1].legend(loc="upper center")
ax[1].set_xlabel("Hours\n(b)")
ax[1].set_ylabel("Energy (kWh)")

format_axes(ax[0])
format_axes(ax[1])
plt.tight_layout()

import os

plt.savefig(os.path.expanduser("~/git/nilm-actionable/figures/hvac/model.pdf"))
plt.savefig(os.path.expanduser("~/git/nilm-actionable/figures/hvac/model.png"))

"""
if SAVE:
    plt.savefig("pred_actual.png")
fig, ax = plt.subplots(nrows=3, sharex=True)
setpoints = [params['t%d' % i].value for i in range(24)]
ax[0].plot(range(24), setpoints)

ax[0].set_ylabel("Predicted setpoint")

#plt.ylim((50, 90))
ax[1].plot(range(24), energy_hourly_mean_df.values)
ax[1].set_ylabel("Hourly mean energy consumption")
ax[2].plot(range(24), temp_hourly_mean_df.values)
ax[2].set_ylabel("Hourly mean temperature")


plt.xlabel("Hour of day")
if SAVE:
    plt.savefig("setpoint.png")
"""