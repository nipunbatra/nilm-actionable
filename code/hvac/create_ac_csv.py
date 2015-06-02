from nilmtk import *
import nilmtk
import forecastio
import datetime
import numpy as np

import pandas as pd
import statsmodels.api as sm
api_key = "122ad90de7a1dfddf34f4c6ba367d828"
lat = 30.25
lng = -97.25

times = []
temps = []
humidity = []
wind_speed = []

"""


start = datetime.datetime(2014, 1, 1)
for offset in range(1, 365):
    print (start+datetime.timedelta(offset))
    forecast = forecastio.load_forecast(api_key, lat, lng, time=start+datetime.timedelta(offset), units="us")
    h = forecast.hourly()
    d = h.data
    for p in d:
        times.append(p.time)
        temps.append(p.d['temperature'])
        humidity.append(p.d['humidity'])
        #wind_speed.append(p.d['windSpeed'])

df = pd.DataFrame({"temp":temps, "humidity":humidity}, index=times)

df = df.tz_localize("Asia/Kolkata")
df = df.tz_convert("US/Central")

"""

df = pd.HDFStore("temp_austin.h5")['/temp']

ds = DataSet("/Users/nipunbatra/Downloads/wikienergy-2.h5")
acs = nilmtk.global_meter_group.select_using_appliances(type='air conditioner')

# For now saving only the first AC per home

# Select start date as 1st of April
start_date = "2014-05-10"

for ac in acs.meters[:5]:
    print ac.instance()
    if ac.appliances[0].metadata['instance'] == 1:
        building_num = ac.building()
        power = ac.load().next()[('power', 'active')]
        on_off_ac = power > 500
        on_off_ac_resampled = on_off_ac.resample("1H", how="sum")
        on_off_ac_resampled = on_off_ac_resampled[start_date:]

        # Find the intersection of the weather and on-off data
        index_intersection = df.index.intersection(on_off_ac_resampled.index)

        df_restricted = df.ix[index_intersection]
        on_off_ac_resampled_restricted = on_off_ac_resampled.ix[
            index_intersection]

        out = index_intersection.hour
        out_reshaped = np.zeros((len(out), 24))

        for i, hour in enumerate(out):
            out_reshaped[i, hour] = 1

        hour_cols = ["h" + str(i) for i in range(24)]
        hour_usage_df = pd.DataFrame(out_reshaped, index=index_intersection)
        hour_usage_df.columns = hour_cols
        hour_usage_df["temp"] = df_restricted["temp"]
        hour_usage_df["humidity"] = df_restricted["humidity"]

        mins_used = on_off_ac_resampled_restricted

        mins_used.to_hdf("temp_hvac.h5", str(building_num) + "_Y")
        hour_usage_df.to_hdf("temp_hvac.h5", str(building_num) + "_X")

        mod = sm.OLS(mins_used, hour_usage_df)

        res = mod.fit()

        print res.summary()
