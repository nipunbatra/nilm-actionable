import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit

WEATHER_HVAC_STORE = "weather_hvac.h5"
building_num = 10

st = pd.HDFStore(WEATHER_HVAC_STORE)

energy = st[str(building_num) + "_Y"]
energy = energy.div(60 * 1000)
hour_usage_df = st[str(building_num) + "_X"]

header_known = hour_usage_df.columns.tolist()
header_unknown = ["t%d" % i for i in range(24)]
header_unknown.append("a1")
header_unknown.append("a2")
header_unknown.append("a3")


def fitFunc(x, t0, t1, t2, t3, t4, t5, t6, t7, t8, t9, t10,
            t11, t12, t13, t14, t15, t16, t17, t18, t19,
            t20, t21, t22, t23, a1, a2, a3):

    return a1 * (
        ((x[24] - t0) * x[0]) +
        ((x[24] - t1) * x[1]) +
        ((x[24] - t2) * x[2]) +
        ((x[24] - t3) * x[3]) +
        ((x[24] - t4) * x[4]) +
        ((x[24] - t5) * x[5]) +
        ((x[24] - t6) * x[6]) +
        ((x[24] - t7) * x[7]) +
        ((x[24] - t8) * x[8]) +
        ((x[24] - t9) * x[9]) +
        ((x[24] - t10) * x[10]) +
        ((x[24] - t11) * x[11]) +
        ((x[24] - t12) * x[12]) +
        ((x[24] - t13) * x[13]) +
        ((x[24] - t14) * x[14]) +
        ((x[24] - t15) * x[15]) +
        ((x[24] - t16) * x[16]) +
        ((x[24] - t17) * x[17]) +
        ((x[24] - t18) * x[18]) +
        ((x[24] - t19) * x[19]) +
        ((x[24] - t20) * x[20]) +
        ((x[24] - t21) * x[21]) +
        ((x[24] - t22) * x[22]) +
        ((x[24] - t23) * x[23])

    ) + a2 * x[25] + a3 * x[26]


def func(x, p):
    # Building the data for a1
    a1_num = 0
    temp = x[header_known.index("temperature")]
    print header_known.index("temperature")
    humidity = x[header_known.index("humidity")]
    windSpeed = x[header_known.index("windSpeed")]
    print header_unknown.index("t%d" % 0)
    print header_known.index("h%d" % 0)
    a1_num = (temp - p[0]
              ) * x[0]
    """
    for i in range(24):
        a1_num += (temp - p[header_unknown.index("t%d" % i)]
                   ) * x[header_known.index("h%d" % i)]
    """

    overall_out = 0
    overall_out += p[header_unknown.index("a1")] * a1_num
    overall_out += p[header_unknown.index("a2")] * humidity
    overall_out += p[header_unknown.index("a3")] * windSpeed
    return overall_out


initial_params = [65 for x in range(24)]
for i in range(3):
    initial_params.append(100)

fitParams, fitCovariances = curve_fit(
    fitFunc, hour_usage_df.T.values, energy.values, p0=initial_params)

predicted = fitFunc(hour_usage_df.T.values, *fitParams)
residual = fitFunc(hour_usage_df.T.values, *fitParams) - energy.values

plt.plot(predicted, label="predicted")
plt.plot(energy, label="gt")
plt.legend()
# plt.show()
#plt.bar(range(24), fitParams[:24])
plt.show()
