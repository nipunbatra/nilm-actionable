from nilmtk import *
import nilmtk
import forecastio
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

# Paramaters for downloading weather data
api_key = "122ad90de7a1dfddf34f4c6ba367d828"
lat = 30.25
lng = -97.25
weather_params = ["temperature", "humidity", "windSpeed"]
start = datetime.datetime(2014, 1, 5)
number_days = 65

# Paramaters for electricity data
start_date = "2014-05-10"

# Does data exist in HDFStore already?
USE_HDF_STORE = True

# Weather data store
WEATHER_DATA_STORE = "weather_austin.h5"

# Weather and HVAC data store
WEATHER_HVAC_STORE = "weather_hvac.h5"


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
    weather_data_df = pd.HDFStore(WEATHER_DATA_STORE)["/temp"]
else:
    weather_data_df = download_weather_data(api_key, lat, lng, weather_params,
                                            start, number_days)
    st = pd.HDFStore(WEATHER_DATA_STORE)
    st["temp"] = weather_data_df


ds = DataSet("/Users/nipunbatra/Downloads/wikienergy-2.h5")

# Selecting all air conditioners in the data set
acs = nilmtk.global_meter_group.select_using_appliances(type='air conditioner')

# For now saving only the first AC per home


for ac in acs.meters[12:14]:
    if ac.appliances[0].metadata['instance'] == 1:
        building_num = ac.building()
        power = ac.load().next()[('power', 'active')]
        energy = power.resample("1H", how="sum")
        # Energy in kWh
        energy = energy.div(60 * 1000)
        energy = energy[start_date:]
        energy.name = "energy"
        # Find the intersection of the weather and on-off data
        index_intersection = weather_data_df.index.intersection(energy.index)

        weather_data_df_restricted = weather_data_df.ix[index_intersection]
        energy_restricted = energy.ix[index_intersection]

        out = index_intersection.hour
        out_reshaped = np.zeros((len(out), 24))

        for i, hour in enumerate(out):
            out_reshaped[i, hour] = 1

        hour_cols = ["h" + str(i) for i in range(24)]
        hour_usage_df = pd.DataFrame(out_reshaped, index=index_intersection)
        hour_usage_df.columns = hour_cols

        for param in weather_params:
            hour_usage_df[param] = weather_data_df_restricted[param]

        energy_restricted.to_hdf(WEATHER_HVAC_STORE, str(building_num) + "_Y")
        hour_usage_df.to_hdf(WEATHER_HVAC_STORE, str(building_num) + "_X")

        # Fitting a linear regression
        mod = sm.OLS(energy_restricted, hour_usage_df)
        res = mod.fit()

        # Getting the regression parameters
        reg_params = res.params
        hour_weather_coeff = reg_params[hour_cols]

        # Stuff for generating plots wrt weather conditions, energy usage and
        # predicted parameters

        hour_weather_coeff.index = [int(x[1:])
                                    for x in hour_weather_coeff.index.values]
        energy_df = pd.DataFrame(energy_restricted.copy())
        energy_hourly_mean_df = get_hourly_aggregate(energy_df)
        energy_hourly_df = get_hourly_data(energy_df, ["energy"])
        weather_mean_hourly = get_hourly_aggregate(weather_data_df_restricted)
        fig, ax = plt.subplots(ncols=1, nrows=5, sharex=True)
        for i, param in enumerate(weather_params):
            weather_mean_hourly[param].plot(
                ax=ax[i], title=param + " " +
                str(round(reg_params[param], 3)))
        hour_weather_coeff.plot(ax=ax[3], title="Hour coefficients")
        energy_hourly_df.plot(
            ax=ax[4], title="energy consumption",
            legend=False, alpha=0.4, style='k-')

        for i in range(5):
            ax[i].set_xlabel("")
        plt.suptitle("R squared:" + str(round(res.rsquared, 3)))
        plt.tight_layout()
        plt.savefig(str(building_num) + ".png")
        plt.show()
