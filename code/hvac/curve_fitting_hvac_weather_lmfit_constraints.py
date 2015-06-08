import os

import numpy as np
import matplotlib.pyplot as plt

plt.style.use('ggplot')
import pandas as pd
from lmfit import minimize, Parameters, report_fit


def get_hourly_aggregate(df, how='mean'):
    df_c = df.copy()
    df_c["hour"] = df_c.index.hour
    return df_c.groupby("hour").mean()


WEATHER_HVAC_STORE = "/Users/nipunbatra/git/nilm-actionable/data/hvac/weather_hvac.h5"
assert os.path.isfile(WEATHER_HVAC_STORE), "File does not exist"

building_num = 115

st = pd.HDFStore(WEATHER_HVAC_STORE)

energy = st[str(building_num) + "_Y"]
#energy = energy.div(60 * 1000)
hour_usage_df = st[str(building_num) + "_X"]

header_known = hour_usage_df.columns.tolist()
header_unknown = ["t%d" % i for i in range(24)]
header_unknown.append("a1")
header_unknown.append("a2")
header_unknown.append("a3")


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
    setpoints = np.array([v['t'+str(i )] for i in range(24)])
    return np.square(model1 + model2 + model3 - data) + 0.01*np.std(setpoints)


# create a set of Parameters
params = Parameters()
for i in range(24):
    params.add('t%d' % i, value=65, min=60, max=80)
for i in range(1, 4):
    params.add('a%d' % i, value=1)

x = hour_usage_df.T.values
data = energy.values
result = minimize(fcn2min, params, args=(x, data))
final = data + result.residual

final = data + result.residual

# write error report
report_fit(params)

SAVE = False

setpoints = [params['t%d' % i].value for i in range(24)]
energy_df = pd.DataFrame({"energy": energy})
energy_hourly_mean_df = get_hourly_aggregate(energy_df)
temp_hourly_mean_df = get_hourly_aggregate(hour_usage_df[["temperature"]])

fig, ax = plt.subplots()
plt.plot(data, label='actual')
plt.plot(final, label='predicted')
plt.legend()
plt.xlabel("Hours")
plt.ylabel("Energy in kWh")
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
