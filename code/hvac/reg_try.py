import numpy as np
from scipy.optimize import curve_fit


def fitFunc(x, t0, t1, t2, t3, t4, t5, t6, t7, t8, t9, t10,
            t11, t12, t13, t14, t15, t16, t17, t18, t19,
            t20, t21, t22, t23, a1, a2, a3):

    return a1 * (
        ((x[24] - t0) * x[0]) +
        ((x[24] - t1) * x[1])
    ) + a2 * x[25] + a3 * x[26]


p0 = [5.11, 3.9, 5.3, 2]

fitParams, fitCovariances = curve_fit(fitFunc, x_3d, x_3d[1, :], p0)
print ' fit coefficients:\n', fitParams
