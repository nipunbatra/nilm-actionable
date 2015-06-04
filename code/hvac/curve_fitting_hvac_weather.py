import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit

WEATHER_HVAC_STORE = "weather_hvac.h5"
building_num = 108

st = pd.HDFStore(WEATHER_HVAC_STORE)

energy = st[str(building_num) + "_Y"]
hour_usage_df = st[str(building_num) + "_X"]

header_known = hour_usage_df.columns.tolist()
header_unknown = ["t%d" % i for i in range(24)]
header_unknown.append("a1")
header_unknown.append("a2")
header_unknown.append("a3")


def func(x, p):
    # Building the data for a1
    a1_num = 0
    temp = x[header_known.index("temperature")]
    humidity = x[header_known.index("humidity")]
    windSpeed = x[header_known.index("windSpeed")]
    for i in range(24):
        a1_num += (temp - p[header_unknown.index("t%d" % i)]
                   ) * x[header_known.index("h%d" % i)]

    overall_out = 0
    overall_out += p[header_unknown.index("a1")] * a1_num
    overall_out += p[header_unknown.index("a2")] * humidity
    overall_out += p[header_unknown.index("a3")] * windSpeed
    return overall_out
