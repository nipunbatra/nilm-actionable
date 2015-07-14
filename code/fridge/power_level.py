import pandas as pd
import os
import glob

script_path = os.path.dirname(os.path.realpath(__file__))

DATA_PATH = os.path.join(script_path, "..","bash_runs_fridge/")
RESULT_PATH = os.path.join(script_path, "..","..","data/fridge")

FOLDER_NAMES = ["N2_K4_T50_Hart", "N2_K4_T50_CO", "N2_K4_T50_FHMM", "N2_K4_T50_CO" , "N3_K4_T50_FHMM"]

out = {}
for folder in FOLDER_NAMES:
    folder_path = os.path.join(DATA_PATH, folder)
    algo_name = folder.split("_")[-1]
    dictionary_key = folder[:3]+algo_name
    out[dictionary_key] = {}
    homes = glob.glob(folder_path+"/*.h5")[:5]
    for home in homes:
        home_number = home.split("/")[-1].split(".")[0]
        with pd.HDFStore(home) as st:
            df = st['/disag']

            ser = df[algo_name]
            ser_pos = ser[ser>20]
            out[dictionary_key][home] = pd.value_counts(ser_pos).head(1).index[0]


