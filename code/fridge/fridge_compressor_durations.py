import warnings

import matplotlib.pyplot as plt
from nilmtk import DataSet
import nilmtk
import pandas as pd
import numpy as np

warnings.filterwarnings("ignore")
from hmmlearn import hmm

ds = DataSet("/Users/nipunbatra/Downloads/wikienergy-2.h5")
fridges = nilmtk.global_meter_group.select_using_appliances(type='fridge')


Wm_to_kwh = 1.66666667* 1e-5


compressor_powers = {
1 : [80, 140],
2 : [80, 140],
8:  [100, 400],
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
68: [80, 200],
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
131:[50,350],
144: [50, 300],
159:[50, 350]

}

defrost_power = {
1: 350,
2: 400,
8: 550,
11: 400,
13: 600, 
14: 400,
15: 500,
18:350,
22: 500,
25:410,
29: 150,
33: 250,
34: 500,
35: 300,
37:600,
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
68: 400,
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
131:400,
144: 600,
159:400
}

def find_on_off(arr):
    diff_arr = np.diff(arr)
    offs_indices = np.where(diff_arr == -1)[0]
    ons_indices = np.where(diff_arr == 1)[0]
    if offs_indices[0]<ons_indices[0]:
        offs_indices = offs_indices[1:]
    l = min(len(ons_indices), len(offs_indices))
    offs_indices = offs_indices[:l]
    ons_indices = ons_indices[:l]
    return ons_indices, offs_indices

"""
def find_on_off_slow(arr):
    i=1
    while i<len(arr):
        if arr[i] - arr[i-1]==1:
            start_index.append(i)
            i = i+1
            #On now, wait till off found
            while i<len(arr) and ((arr[i] - arr[i-1])!=-1) :
                i = i+1
            if i<len(arr):
                stop_index.append(i)
        else:
            i = i+1
    l = len(stop_index)
    start_index = start_index[:l]

"""

def find_compressor_defrost(n):
    df = fridges.meters[n].load().next()[('power','active')]
    [compressor_min, compressor_max] = compressor_powers[n]
    defrost_min = defrost_power[n]
    compressor = (df>compressor_min) & (df<compressor_max)
    defrost_idx = df>defrost_min
    defrost = defrost_idx
    compressor[defrost_idx] = False
    
    #return compressor
    # Eliminate 1 minute cycles
    for i in range(len(df)-2):
        if compressor.ix[i]== False and compressor.ix[i+1] == True and compressor.ix[i+2]==False:
            compressor.ix[i+1] = False
        elif compressor.ix[i]== True and compressor.ix[i+1] == False and compressor.ix[i+2]==True:
            compressor.ix[i+1] = True
    return compressor, defrost


def compute_fractions_df(df):
    a, b, c, tot, mins = fractions_df(n)
    return wm_to_kwh_per_month(tot, mins), wm_to_kwh_per_month(a, mins), wm_to_kwh_per_month(c, mins), wm_to_kwh_per_month(b, mins)


def compute_fractions(n):
    a, b, c, tot, mins = fractions(n)
    return wm_to_kwh_per_month(tot, mins), wm_to_kwh_per_month(a, mins), wm_to_kwh_per_month(c, mins), wm_to_kwh_per_month(b, mins)



def wm_to_kwh_per_month(wm, mins):
    return wm*Wm_to_kwh/(mins*1.0/(1440*30))


def return_states_df(n):
    df = fridges.meters[n].load().next()[('power','active')]
    X = df.head(10000)
    X = X[X<2000]
    X = X.reshape((len(X),1))
    # Defrost state? (N=3), else 2
    model = hmm.GaussianHMM(n_components=2, covariance_type="full", n_iter=1000)
    model.fit([X])
    z = model.means_.reshape((2,))
    z.sort()
    raw_power = df.values
    y1 = np.abs(raw_power - z[0])
    y2 = np.abs(raw_power - z[1])
    y_act = np.zeros(y1.shape)
    y_act[np.where((y1-y2>0))[0]]=1
    #y_act[np.where((y1-y2>0)&(y2<4*z[1]))[0]]=1
    df_states = pd.Series(y_act, index=df.index)
    return df_states, z[1]

def fractions(n):
    f = fridges.meters[n].load().next()[('power','active')]
    c, d = find_compressor_defrost(n)
    power_c_sum = f[c].sum()
    print power_c_sum
    
    df_cm, df_d = find_on_off_durations(n)
    baseline = df_cm.between_time("01:00", "05:00").median()
    baseline_duty_percent = baseline['on']/(baseline['on']+baseline['off'])
    
    print baseline_duty_percent
    
    total_mins = len(f)
    baseline_energy = total_mins*baseline_duty_percent*f[c].mean()
    
    print total_mins
    
    defrost_energy_self = f[d].sum()
    defrost_energy_extra_compressor = 0.0
    for i in range(len(df_d.index)):
        runtime = df_cm[df_d.index[i]:].head(3)['on'].max()
        if runtime>baseline['on']:
            extra_run_energy = (runtime-baseline['on'])*f[c].mean()
            defrost_energy_extra_compressor = defrost_energy_extra_compressor +extra_run_energy
            power_c_sum = power_c_sum - extra_run_energy
    defrost_energy = defrost_energy_self + defrost_energy_extra_compressor
    
    usage_energy = power_c_sum - baseline_energy
    total_energy = f.sum()
    return baseline_energy, usage_energy, defrost_energy, total_energy, total_mins

def return_states_df_defrost(n):
    df = fridges.meters[n].load().next()[('power','active')]
    X = df.head(10000)
    X = X[X<2000]
    X = X.reshape((len(X),1))
    # Defrost state? (N=3), else 2
    model = hmm.GaussianHMM(n_components=3, covariance_type="full", n_iter=1000)
    model.fit([X])
    z = model.means_.reshape((3,))
    z.sort()
    raw_power = df.values
    p = model.predict(raw_power.reshape((len(raw_power),1)))
    y1 = np.abs(raw_power - z[0])
    y2 = np.abs(raw_power - z[1])
    y_act = np.zeros(y1.shape)
    y_act[np.where((y1-y2>0)&(y2<1.2*z[1]))[0]]=1
    df_states = pd.Series(y_act, index=df.index)
    df_states_hmm = pd.Series(p, index=df.index)
    return df_states, df_states_hmm, z



def find_weekend_indices(datetime_array):
    indices=[]
    for i in range(len(datetime_array)):
        if datetime_array[i].weekday()>=5:
            indices.append(i)
    return indices

def highlight_weekend(weekend_indices,ax):
    i=0
    while i<len(weekend_indices):
         ax.axvspan(weekend_indices[i], weekend_indices[i]+2, facecolor='green', edgecolor='none', alpha=.2)
         i+=2

def find_on_off_durations(n):
    c, d = find_compressor_defrost(n)
    on_c, off_c = find_on_off(c.astype('int').values)
    on_d, off_d = find_on_off(d.astype('int').values)
    to_ignore =[]

    # We now need to remove the extra run of compressor due to defrost.
    # We look for defrost off and ignore the next compressor cycle 

    for defrost_off_index in off_d:
      next_compressor_index = np.where(on_c>defrost_off_index)[0][0]
      to_ignore.append(next_compressor_index)
      to_ignore.append(next_compressor_index+1)
      to_ignore.append(next_compressor_index+2)
      to_ignore.append(next_compressor_index-1)


    on_duration_compressor = pd.DataFrame({"on":(off_c-on_c)[:-1], 
                                    "off":on_c[1:] - off_c[:-1]}, index=c.index[on_c[:-1]]).sort_index()

    to_consider = [x for x in range(len(on_duration_compressor)) if x not in to_ignore]

    on_duration_compressor_filtered = on_duration_compressor.ix[to_consider]

    on_duration_defrost = pd.DataFrame({"on":(off_d-on_d)[:-1], 
                                "off":on_d[1:] - off_d[:-1]}, index=d.index[on_d[:-1]]).sort_index()
    on_duration_defrost = on_duration_defrost[on_duration_defrost.on>10]
    
    return on_duration_compressor_filtered, on_duration_defrost

def find_on_off_durations_with_without_filter(n):
    c, d = find_compressor_defrost(n)
    on_c, off_c = find_on_off(c.astype('int').values)
    on_d, off_d = find_on_off(d.astype('int').values)
    to_ignore =[]

    # We now need to remove the extra run of compressor due to defrost.
    # We look for defrost off and ignore the next compressor cycle 

    for defrost_off_index in off_d:
      next_compressor_index = np.where(on_c>defrost_off_index)[0][0]
      to_ignore.append(next_compressor_index)
      to_ignore.append(next_compressor_index+1)
      to_ignore.append(next_compressor_index+2)
      to_ignore.append(next_compressor_index-1)


    on_duration_compressor = pd.DataFrame({"on":(off_c-on_c)[:-1], 
                                    "off":on_c[1:] - off_c[:-1]}, index=c.index[on_c[:-1]]).sort_index()

    to_consider = [x for x in range(len(on_duration_compressor)) if x not in to_ignore]

    on_duration_compressor_filtered = on_duration_compressor.ix[to_consider]

    on_duration_defrost = pd.DataFrame({"on":(off_d-on_d)[:-1], 
                                "off":on_d[1:] - off_d[:-1]}, index=d.index[on_d[:-1]]).sort_index()
    on_duration_defrost = on_duration_defrost[on_duration_defrost.on>10]
    
    return on_duration_compressor, on_duration_compressor_filtered, on_duration_defrost





def find_baseline(n):
    df_c, df_d = find_on_off_durations(n)
    times=df_c.index
    return df_c.groupby([times.hour]).median().min()




def execute():
    o = {}
    for n in compressor_powers.keys()[:]:
        if n not in o.keys():
            print n
            try:
                o[n] = compute_fractions(n)
            except:
                pass
    d = pd.DataFrame(o).T
    d.columns = ["total", "baseline", "defrost", "usage"]

    dp  = d[d.usage>0]
    dp["artifical_sum"] = dp.baseline+dp.defrost+dp.usage

    dp["baseline_percentage"] = dp.baseline*100/dp.artifical_sum
    dp["defrsot_percentage"] = dp.defrost*100/dp.artifical_sum
    dp["usage_percentage"] = dp.usage*100/dp.artifical_sum

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







    out = {}
    for n in compressor_powers.keys()[:]:
        if n not in out.keys():
            print n
            try:
                t = find_baseline(n)
                out[n] = 1.0*t.on/(t.on+t.off)
            except:
                pass


fig, ax = plt.subplots(nrows=2, ncols=4)
count = 0
binwidth=10
for n in compressor_powers.keys()[:8]:
    print n
    c, cf, d = find_on_off_durations_with_without_filter(n)
    maximum = max(c.on.max(), cf.on.max())
    minimum = min(c.on.min(), cf.on.min())
    bins=np.arange(minimum, maximum + binwidth, binwidth)
    c.on.hist(bins=bins, ax=ax[count/4][count%4])
    cf.on.hist(alpha=0.5, bins=bins, ax=ax[count/4][count%4], facecolor='r')
    ax[count/4][count%4].set_title("Fridge %d\nNumber of defrost cycles:%d " %(n, len(d)))

    count = count+1
plt.tight_layout()




if n not in out.keys():
    print n
    try:
        t = find_baseline(n)
        out[n] = 1.0*t.on/(t.on+t.off)
    except:
        pass






