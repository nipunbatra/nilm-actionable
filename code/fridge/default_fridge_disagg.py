import warnings

from nilmtk import DataSet
import nilmtk
import pandas as pd
import numpy as np
import os
warnings.filterwarnings("ignore")


ds = DataSet(os.path.expanduser("~/Downloads/wikienergy-2.h5"))
fridges = nilmtk.global_meter_group.select_using_appliances(type='fridge')

Wm_to_kwh = 1.66666667 * 1e-5

compressor_powers = {
    1: [80, 140],
    2: [80, 140],
    8: [100, 400],
    11: [90, 350],
    13: [100, 160],
    14: [70, 200],
    15: [100, 200],
    18: [50, 200],
    22: [50, 400],
    25: [50, 400],
    29: [50, 120],
    33: [50, 200],
    34: [50, 400],
    35: [100, 200],
    37: [100, 200],
    42: [50, 200],
    43: [50, 200],
    44: [50, 200],
    45: [50, 200],
    46: [50, 350],
    47: [50, 200],
    50: [100, 200],
    51: [100, 150],
    52: [100, 200],
    54: [80, 160],
    55: [60, 150],
    57: [80, 200],
    58: [80, 250],
    59: [80, 150],
    60: [80, 150],
    61: [80, 150],
    67: [80, 300],
    68: [50, 200],
    70: [80, 200],
    72: [80, 300],
    75: [80, 200],
    76: [80, 200],
    78: [80, 400],
    79: [80, 400],
    83: [80, 150],
    84: [80, 400],
    87: [80, 300],
    88: [80, 200],
    89: [100, 160],
    92: [200, 300],
    93: [100, 200],
    95: [100, 220],
    97: [100, 200],
    99: [100, 200],
    100: [100, 200],
    102: [100, 200],
    103: [100, 220],
    104: [200, 300],
    106: [100, 200],
    107: [100, 200],
    109: [100, 250],
    110: [80, 200],
    112: [100, 200],
    114: [100, 200],
    115: [100, 200],
    116: [100, 200],
    118: [80, 200],
    119: [80, 150],
    123: [100, 200],
    124: [100, 200],
    125: [100, 200],
    126: [100, 200],
    128: [100, 200],
    129: [100, 200],
    130: [100, 200],
    131: [50, 350],
    133: [50, 100],
    134: [80, 200],
    135: [100, 200],
    136: [50, 200],
    138: [50, 200],
    139: [50, 200],
    140: [50, 150],
    142: [100, 200],
    144: [50, 300],
    145: [50, 200],
    146: [50, 200],
    149: [100, 200],
    151: [50, 150],
    152: [100, 200],
    153: [100, 250],
    154: [100, 220],
    155: [200, 300],
    157: [100, 250],
    158: [100, 200],
    159: [50, 350],
    161: [50, 200],
    163: [100, 200],
    167: [100, 200],
    169: [100, 250],
    170: [100, 200]

}

defrost_power = {
    1: 350,
    2: 400,
    8: 550,
    11: 400,
    13: 600,
    14: 400,
    15: 500,
    18: 350,
    22: 500,
    25: 410,
    29: 150,
    33: 250,
    34: 500,
    35: 300,
    37: 600,
    42: 200,
    43: 400,
    44: 400,
    45: 400,
    46: 1000,
    47: 400,
    50: 500,
    51: 300,
    52: 400,
    54: 300,
    55: 400,
    57: 350,
    58: 350,
    59: 350,
    60: 400,
    61: 350,
    67: 400,
    68: 300,
    70: 350,
    72: 400,
    75: 400,
    76: 400,
    78: 600,
    79: 400,
    83: 300,
    84: 400,
    87: 300,
    88: 300,
    89: 420,
    92: 450,
    93: 550,
    95: 400,
    97: 600,
    99: 350,
    100: 500,
    102: 550,
    103: 600,
    104: 500,
    106: 350,
    107: 350,
    109: 550,
    110: 400,
    112: 400,
    114: 450,
    115: 400,
    116: 450,
    118: 400,
    119: 450,
    123: 450,
    124: 400,
    125: 400,
    126: 600,
    128: 350,
    129: 600,
    130: 450,
    131: 400,
    133: 280,
    134: 500,
    135: 400,
    136: 350,
    138: 400,
    139: 320,
    140: 300,
    142: 500,
    144: 600,
    145: 400,
    146: 380,
    149: 600,
    150: 600,
    151: 350,
    152: 400,
    153: 400,
    154: 400,
    155: 700,
    157: 400,
    158: 600,
    159: 400,
    161: 350,
    163: 400,
    167: 400,
    169: 500,
    170: 650,
}


def find_on_off(arr):
    diff_arr = np.diff(arr)
    offs_indices = np.where(diff_arr == -1)[0]
    ons_indices = np.where(diff_arr == 1)[0]
    if offs_indices[0] < ons_indices[0]:
        offs_indices = offs_indices[1:]
    l = min(len(ons_indices), len(offs_indices))
    offs_indices = offs_indices[:l]
    ons_indices = ons_indices[:l]
    return ons_indices, offs_indices


def find_compressor_defrost(df, n):
    [compressor_min, compressor_max] = compressor_powers[n]
    defrost_min = defrost_power[n]
    compressor = (df > compressor_min) & (df < compressor_max)
    defrost_idx = df > defrost_min
    defrost = defrost_idx
    compressor[defrost_idx] = False

    # return compressor
    # Eliminate 1 minute cycles
    for i in range(len(df) - 2):
        if compressor.ix[i] is False and compressor.ix[i + 1] is True and compressor.ix[i + 2] is False:
            compressor.ix[i + 1] = False
        elif compressor.ix[i] is True and compressor.ix[i + 1] is False and compressor.ix[i + 2] is True:
            compressor.ix[i + 1] = True
    return compressor, defrost

def compute_fractions(n):
    a, b, c, tot, mins = fractions(n)
    return wm_to_kwh_per_month(tot, mins), wm_to_kwh_per_month(a, mins), wm_to_kwh_per_month(c,
                                                                                             mins), wm_to_kwh_per_month(
        b, mins)

def compute_fractions_new(n):
    a, b, c, tot, mins, usage_cycles, non_usage_cycles, defrost_cycles, baseline_duty_percent = fractions_new(n, 17)
    return wm_to_kwh_per_month(tot, mins), wm_to_kwh_per_month(a, mins), wm_to_kwh_per_month(c,
                                                                                             mins), wm_to_kwh_per_month(
        b, mins), usage_cycles, non_usage_cycles, defrost_cycles, baseline_duty_percent, mins


def wm_to_kwh_per_month(wm, mins):
    return wm * Wm_to_kwh / (mins * 1.0 / (1440 * 30))

def fractions_new(n, percentage_threshold = 17):
    f = fridges.meters[n].load().next()[('power', 'active')]
    c, d = find_compressor_defrost(f, n)
    # Sum of all the power when the compressor was ON
    mean_compressor_power = f[c].mean()

    df_cm, df_d = find_on_off_durations(f, n)
    baseline = df_cm.between_time("01:00", "05:00").median()
    baseline_duty_percent = baseline['on'] / (baseline['on'] + baseline['off'])


    total_mins = len(f)
    baseline_energy = 0.0


    defrost_energy_self = f[d].sum()
    defrost_energy_extra_compressor = 0.0
    for i in range(len(df_d.index)):
        runtime = df_cm[df_d.index[i]:].head(3)['on'].max()
        if runtime > baseline['on']:
            extra_run_energy = (runtime - baseline['on']) * mean_compressor_power
            defrost_energy_extra_compressor = defrost_energy_extra_compressor + extra_run_energy

    defrost_energy = defrost_energy_self + defrost_energy_extra_compressor

    baseline_threshold = baseline_duty_percent + baseline_duty_percent*percentage_threshold/100
    print baseline_threshold, baseline_duty_percent

    df_cm["duty"] = df_cm["on"]*1.0/(df_cm["on"] + df_cm["off"])
    usage_df_cm = df_cm[df_cm.duty> baseline_threshold]
    usage_df_cm["cycle_mins"] =  (usage_df_cm["on"] + usage_df_cm["off"])
    usage_mins_df = (usage_df_cm.duty - baseline_duty_percent)*usage_df_cm["cycle_mins"]
    #usage_mins_df = usage_df_cm.on - (cycle_mins*baseline_duty_percent*1.0/100)
    usage_mins = usage_mins_df.sum()
    baseline_mins = usage_df_cm["cycle_mins"]*baseline_duty_percent
    baseline_energy = baseline_energy + baseline_mins.sum()*mean_compressor_power
    non_usage_df_cm = df_cm[df_cm.duty<= baseline_threshold]
    non_usage_mins = non_usage_df_cm["on"].sum()
    baseline_energy = baseline_energy + non_usage_mins*mean_compressor_power
    usage_energy = usage_mins*mean_compressor_power
    #usage_energy = usage_energy - defrost_energy_extra_compressor
    total_energy = f.sum()
    return baseline_energy, usage_energy, defrost_energy, total_energy,\
           total_mins, len(usage_df_cm), len(non_usage_df_cm), len(df_d), baseline_duty_percent


def fractions(n):
    f = fridges.meters[n].load().next()[('power', 'active')]
    c, d = find_compressor_defrost(f, n)
    # Sum of all the power when the compressor was ON
    power_c_sum = f[c].sum()

    df_cm, df_d = find_on_off_durations(f, n)
    baseline = df_cm.between_time("01:00", "05:00").median()
    baseline_duty_percent = baseline['on'] / (baseline['on'] + baseline['off'])

    print baseline_duty_percent

    total_mins = len(f)
    baseline_energy = total_mins * baseline_duty_percent * f[c].mean()

    print total_mins

    defrost_energy_self = f[d].sum()
    defrost_energy_extra_compressor = 0.0
    for i in range(len(df_d.index)):
        runtime = df_cm[df_d.index[i]:].head(3)['on'].max()
        if runtime > baseline['on']:
            extra_run_energy = (runtime - baseline['on']) * f[c].mean()
            defrost_energy_extra_compressor = defrost_energy_extra_compressor + extra_run_energy
            power_c_sum = power_c_sum - extra_run_energy
    defrost_energy = defrost_energy_self + defrost_energy_extra_compressor

    usage_energy = power_c_sum - baseline_energy
    total_energy = f.sum()
    return baseline_energy, usage_energy, defrost_energy, total_energy, total_mins


def find_on_off_durations(power_df, n):
    c, d = find_compressor_defrost(power_df, n)
    c_array = c.astype('int').values.reshape(len(c),)
    d_array = d.astype('int').values.reshape(len(c),)

    on_c, off_c = find_on_off(c_array)
    on_d, off_d = find_on_off(d_array)
    to_ignore = []

    """

    # We now need to remove the extra run of compressor due to defrost.
    # We look for defrost off and ignore the next compressor cycle 

    for defrost_off_index in off_d:
        next_compressor_index = np.where(on_c > defrost_off_index)[0][0]
        to_ignore.append(next_compressor_index)
        to_ignore.append(next_compressor_index + 1)
        to_ignore.append(next_compressor_index + 2)
        to_ignore.append(next_compressor_index - 1)

    """
    to_ignore = []

    on_duration_compressor = pd.DataFrame({"on": (off_c - on_c)[:-1],
                                           "off": on_c[1:] - off_c[:-1]},
                                          index=c.index[on_c[:-1]]).sort_index()

    to_consider = [x for x in range(len(on_duration_compressor)) if x not in to_ignore]

    on_duration_compressor_filtered = on_duration_compressor.ix[to_consider]

    on_duration_defrost = pd.DataFrame({"on": (off_d - on_d)[:-1],
                                        "off": on_d[1:] - off_d[:-1]},
                                       index=d.index[on_d[:-1]]).sort_index()
    on_duration_defrost = on_duration_defrost[on_duration_defrost.on > 10]

    return on_duration_compressor_filtered, on_duration_defrost


def find_on_off_durations_with_without_filter(n):
    c, d = find_compressor_defrost(n)
    on_c, off_c = find_on_off(c.astype('int').values)
    on_d, off_d = find_on_off(d.astype('int').values)
    to_ignore = []

    # We now need to remove the extra run of compressor due to defrost.
    # We look for defrost off and ignore the next compressor cycle 

    for defrost_off_index in off_d:
        next_compressor_index = np.where(on_c > defrost_off_index)[0][0]
        to_ignore.append(next_compressor_index)
        to_ignore.append(next_compressor_index + 1)
        to_ignore.append(next_compressor_index + 2)
        to_ignore.append(next_compressor_index - 1)

    on_duration_compressor = pd.DataFrame({"on": (off_c - on_c)[:-1],
                                           "off": on_c[1:] - off_c[:-1]}, index=c.index[on_c[:-1]]).sort_index()

    to_consider = [x for x in range(len(on_duration_compressor)) if x not in to_ignore]

    on_duration_compressor_filtered = on_duration_compressor.ix[to_consider]

    on_duration_defrost = pd.DataFrame({"on": (off_d - on_d)[:-1],
                                        "off": on_d[1:] - off_d[:-1]}, index=d.index[on_d[:-1]]).sort_index()
    on_duration_defrost = on_duration_defrost[on_duration_defrost.on > 10]

    return on_duration_compressor, on_duration_compressor_filtered, on_duration_defrost


def find_baseline(n):
    df_c, df_d = find_on_off_durations(n)
    times = df_c.index
    return df_c.groupby([times.hour]).median().min()



def execute():
    import time
    start = time.time()
    o = {}
    disag_dict = {}
    o_new = {}
    #for n in [68]:
    for n in compressor_powers.keys()[:]:
        if n not in o.keys():
            print "Computing for", n
            try:
                #o[n] = compute_fractions(n)
                disag_dict[n] = disag(n)

            except Exception as e:
                print "EXCEPTION"
                print e
    end = time.time()

    #d = pd.DataFrame(o).T
    #d.columns = ["total", "baseline", "defrost", "usage"]

    d_new = pd.DataFrame(o_new).T
    d_new.columns = ["total", "baseline", "defrost", "usage",
                     "usage_cycles", "non_usage_cycles",
                     "defrost_cycles","baseline_duty_percent", "total_mins"]


    #d = d[d.usage > 0]
    d_new["artifical_sum"] = d_new.baseline + d_new.defrost + d_new.usage

    d_new["baseline_percentage"] = d_new.baseline * 100 / d_new.total
    d_new["defrost_percentage"] = d_new.defrost * 100 / d_new.total
    d_new["usage_percentage"] = d_new.usage * 100 / d_new.total

    d_new.to_csv("../../data/fridge/usage_defrost_cycles.csv", index_label="home")


"""
ds = DataSet("/Users/nipunbatra/Downloads/wikienergy-2.h5")

original_name_dict = {b.metadata['original_name']:b.identifier.instance for b in ds.buildings.values()}
original_name_map = pd.Series(original_name_dict)
reverse_name_map = pd.Series({v:k for k,v in original_name_dict.iteritems() })

fridges = nilmtk.global_meter_group.select_using_appliances(type='fridge')
fridges_dict_original = {i:ds.buildings[fridges.meters[i].building()].metadata['original_name'] for i in range(len(fridges.meters))}
fridges_dict_nilmtk = {i:fridges.meters[i].building() for i in range(len(fridges.meters))}
fridges_map_original = pd.Series(fridges_dict_original)
fridges_map_nilmtk = pd.Series(fridges_dict_nilmtk)
to_ignore = [0, 3, 4, 5, 6, 7, 9, 10, 12, 16, 17, 19, 20, 21, 23, 24, 27, 30, 31,
             32, 36, 38, 39, 40, 41, 53, 54, 58, 73, 74, 77, 82, 85, 86, 90, 91 ,
             94, 95, 96, 98, 99, 101, 117, 119, 121, 122, 125, 127, 133, 137,
             141, 147, 156, 157, 160, 165, 166, 170, 171, 172]
maybe = [60, 80, 81, 105, 113, 120, 126, 159, 162, 14, 46]
anomaly = [6, 48]

"""




"""
out = {}
for n in compressor_powers.keys()[:]:
    if n not in out.keys():
        print n
        try:
            t = find_baseline(n)
            out[n] = 1.0*t.on/(t.on+t.off)
        except:
            pass

"""
