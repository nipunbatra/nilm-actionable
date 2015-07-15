import datetime

import forecastio
import numpy as np
import pandas as pd



# Paramaters for downloading weather data
api_key = "122ad90de7a1dfddf34f4c6ba367d828"
lat = 30.25
lng = -97.25
weather_params = ["temperature", "humidity", "windSpeed"]
start = datetime.datetime(2013, 5, 1)
number_days = 120

# Paramaters for electricity data
start_date = "2013-1-5"

START, STOP = '2013-07-01', '2013-07-31'

# Does data exist in HDFStore already?
USE_HDF_STORE = False

# Weather data store
WEATHER_DATA_STORE = "../../data/hvac/weather_2013.h5"

# Weather and HVAC data store
WEATHER_HVAC_STORE = "../../data/hvac/weather_hvac_2013.h5"


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


def download_weather_data(api_key, lat, lng,
                          weather_params,
                          start, number_days):

    times = []
    out = {k: [] for k in weather_params}
    for offset in range(1, number_days):
        print(start + datetime.timedelta(offset))
        forecast = forecastio.load_forecast(
            api_key, lat, lng, time=start + datetime.timedelta(offset),
            units="us")
        h = forecast.hourly()
        d = h.data
        for p in d:
            times.append(p.time)
            for param in weather_params:
                if param in p.d:
                    out[param].append(p.d[param])
                else:
                    out[param].append(np.NaN)

    df = pd.DataFrame(out, index=times)
    df = df.fillna(method='ffill')

    df = df.tz_localize("Asia/Kolkata")
    df = df.tz_convert("US/Central")
    return df

if USE_HDF_STORE:
    weather_data_df = pd.HDFStore(WEATHER_DATA_STORE)["/weather"]
else:
    weather_data_df = download_weather_data(api_key, lat, lng, weather_params,
                                            start, number_days)
    st = pd.HDFStore(WEATHER_DATA_STORE)
    st["weather"] = weather_data_df

sys.exit(0)

store = pd.HDFStore("/Users/nipunbatra/Downloads/wiki-temp.h5")

a = store.keys()
def num_from_key(key):
    return int(key[1:])

def key_from_num(num):
    return "/"+str(num)


ids = map(num_from_key, a)

cols = ['programmable_thermostat_currently_programmed',
        'temp_summer_weekday_workday', 'temp_summer_weekday_morning',
        'temp_summer_weekday_evening', 'temp_summer_sleeping_hours_hours']
from copy import deepcopy
cols_plus_data_id = deepcopy(cols)
cols_plus_data_id.insert(0, "dataid")


df = pd.read_csv("../../data/total/survey_2013.csv")

df = df[cols_plus_data_id].dropna()
survey_homes = df.dataid.values
data_homes = np.array(ids)


data_homes = np.array(ids)

ids_common = np.intersect1d(survey_homes, data_homes)

ids_common_exist = []
for id_home in ids_common[:]:
    try:
        print id_home
        d = store[key_from_num(id_home)]['air1'][START:STOP]
        if len(d) > 0:
            ids_common_exist.append(id_home)
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
                energy_restricted.to_hdf(
                    WEATHER_HVAC_STORE, str(id_home) + "_Y")
                hour_usage_df.to_hdf(WEATHER_HVAC_STORE, str(id_home) + "_X")

    except Exception, e:
        print e
        pass

