import warnings
import sys

from nilmtk import DataSet
import nilmtk
import pandas as pd
import numpy as np
sys.path.append('/Users/nipunbatra/git/nilm-actionable/code/common')

from common_functions import latexify, format_axes

import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

ds = DataSet("/Users/nipunbatra/Downloads/wikienergy-2.h5")
fridges = nilmtk.global_meter_group.select_using_appliances(type='fridge')

Wm_to_kwh = 1.66666667 * 1e-5

fridges_to_use = [153, 58, 165, 60, 54, 177, 81, 42]

compressor_powers_local = {
    153: [100, 200],
    58: [100, 250],
    165: [100, 200],
    60: [80, 200],
    54:[50, 250],
    177:[50, 250],
    81:[50, 350],
    42:[50, 350]
}

defrost_power_local = {
    153: 500,
    58: 600,
    165: 400,
    60:400,
    54: 400,
    177: 400,
    81:400,
    42:400

}

def find_compressor_defrost(df, n):
    [compressor_min, compressor_max] = compressor_powers_local[n]
    defrost_min = defrost_power_local[n]
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

def fractions_new(n, percentage_threshold = 17):
    f = ds.buildings[n].elec[('fridge', 1)].load().next()[('power', 'active')]
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


def compute_fractions_new_ds(n):
    a, b, c, tot, mins, usage_cycles, non_usage_cycles, defrost_cycles, baseline_duty_percent = fractions_new(n, 17)
    return wm_to_kwh_per_month(tot, mins), wm_to_kwh_per_month(a, mins), wm_to_kwh_per_month(c,
                                                                                             mins), wm_to_kwh_per_month(
        b, mins), usage_cycles, non_usage_cycles, defrost_cycles, baseline_duty_percent, mins

Wm_to_kwh = 1.66666667 * 1e-5

def wm_to_kwh_per_month(wm, mins):
    return wm * Wm_to_kwh / (mins * 1.0 / (1440 * 30))

o_new = {}
for n in fridges_to_use:
    print n
    if n not in o_new.keys():
        print "Computing for", n
        try:
            o_new[n] = compute_fractions_new_ds(n)
        except Exception as e:
            print "EXCEPTION"
            print e

#d = pd.DataFrame(o).T
#d.columns = ["total", "baseline", "defrost", "usage"]

d_new = pd.DataFrame(o_new).T
d_new.columns = ["total", "baseline", "defrost", "usage",
                 "usage_cycles", "non_usage_cycles",
                 "defrost_cycles","baseline_duty_percent", "total_mins"]


#d = d[d.usage > 0]
d_new["artifical_sum"] = d_new.baseline + d_new.defrost + d_new.usage

d_new["baseline_percentage"] = d_new.baseline * 100 / d_new.artifical_sum
d_new["defrost_percentage"] = d_new.defrost * 100 / d_new.artifical_sum
d_new["usage_percentage"] = d_new.usage * 100 / d_new.artifical_sum

pairs = [
    [58, 153],
    [60, 165],
    [54, 177],
    [42, 81]
]




latexify()

"""

plot_dict = {}
for part in ["baseline", "defrost", "usage"]:
    plot_dict[part] = {}
    for pair in pairs:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        plot_dict[part][str(pair)] = d_new.ix[pair][part].values
        pd.DataFrame(plot_dict[part]).T.plot(ax=ax, kind='bar', legend=False, rot=0)
        plt.ylabel(str.capitalize(part) + " energy consumption\n per month (kWh)")
        plt.xlabel("Same company fridge pairs")
        format_axes(ax)
        plt.tight_layout()
        plt.savefig("../../figures/fridge/same_fridge_"+part+".png")
        plt.savefig("../../figures/fridge/same_fridge_"+part+".pdf")


"""

idx_sorted = np.array(pairs).flatten()
d_res = d_new.ix[idx_sorted][["baseline", "usage", "defrost"]]

N =len(pairs)

ind = np.arange(N)  # the x locations for the groups
width = 0.35       # the width of the bars

p = {}

fridge_1 = pd.concat([d_res.ix[pairs[i][0]] for i in range(N)], axis=1)
fridge_2 = pd.concat([d_res.ix[pairs[i][1]] for i in range(N)], axis=1)

p1= plt.bar(ind, fridge_1.ix["baseline"].values,   width, color='b')
p2 = plt.bar(ind, fridge_1.ix["usage"].values, width, color='r',
         bottom=fridge_1.ix["baseline"].values)
p3 = plt.bar(ind, fridge_1.ix["defrost"].values, width, color='g',
         bottom=fridge_1.ix["baseline"].values+fridge_1.ix["usage"].values)


plt.bar(ind+width, fridge_2.ix["baseline"].values,   width, color='b')
plt.bar(ind+width, fridge_2.ix["usage"].values, width, color='r',
         bottom=fridge_2.ix["baseline"].values)
plt.bar(ind+width, fridge_2.ix["defrost"].values, width, color='g',
         bottom=fridge_2.ix["baseline"].values+fridge_2.ix["usage"].values)

plt.xticks(ind+width/2., map(str, pairs) )

format_axes(plt.gca())
plt.ylabel("Energy per month in kWh")
plt.xlabel("Home pairs having identical fridges")
plt.legend( (p1[0], p2[0], p3[0]), ('Baseline', 'Usage', 'Defrost'), loc='upper center', bbox_to_anchor=(0.5, 1.2),
          ncol=3)

plt.tight_layout()
plt.savefig("../../figures/fridge/identical_fridges.png", bbox_inches="tight")
plt.savefig("../../figures/fridge/identical_fridges.pdf", bbox_inches="tight")




