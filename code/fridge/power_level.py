import pandas as pd
import os
import glob
import numpy as np
script_path = os.path.dirname(os.path.realpath(__file__))

DATA_PATH = os.path.join(script_path, "..","bash_runs_fridge/")
RESULT_PATH = os.path.join(script_path, "..","..","data/fridge")

FOLDER_NAMES = ["N2_K4_T50_Hart", "N2_K4_T50_CO", "N2_K4_T50_FHMM", "N3_K4_T50_CO" , "N3_K4_T50_FHMM"]

out = {}
out["GT"] = {}
for folder in FOLDER_NAMES:
    folder_path = os.path.join(DATA_PATH, folder)
    algo_name = folder.split("_")[-1]
    dictionary_key = folder[:3]+algo_name
    out[dictionary_key] = {}

    homes = glob.glob(folder_path+"/*.h5")
    for home in homes:
        home_number = home.split("/")[-1].split(".")[0]
        print home_number
        with pd.HDFStore(home) as st:
            df = st['/disag']

            ser = df[algo_name]
            gt = df["GT"]
            gt_pos = gt[gt>20]
            gt_median = gt_pos.median()
            ser_pos = ser[ser>20]
            abs_difference = np.abs(ser_pos.unique()-gt_median)
            ser_pos_closest = ser_pos.unique()[(np.argmin(abs_difference))]
            out[dictionary_key][home_number] = ser_pos_closest
            out["GT"][home_number] = gt_median


df = pd.DataFrame(out)
df = df.dropna()
#df.to_csv()