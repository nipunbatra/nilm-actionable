import os
import pandas as pd
import numpy as np

def find_common_dataids():
    to_consider = ["N2_K3_T50_CO","N3_K3_T50_CO",
                   "N2_K3_T50_FHMM","N3_K3_T50_FHMM",
                   "N2_K3_T50_CO","N2_K3_T50_Hart"]

    script_path = os.path.dirname(os.path.realpath(__file__))
    DATA_PATH = os.path.join(script_path, "../../data/hvac")

    df = pd.read_csv(os.path.join(DATA_PATH, "minutes_GT.csv"))
    df = df.dropna()
    print len(df)
    dataids_overall = df.dataid
    for folder in to_consider:
        df = pd.read_csv(os.path.join(DATA_PATH, "minutes_%s.csv" %folder))
        dataids = df.dataid
        dataids_overall = np.intersect1d(dataids_overall, dataids)
    return dataids_overall



