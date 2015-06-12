import matplotlib.pyplot as plt

plt.style.use('ggplot')
from nilmtk import DataSet
import nilmtk
import warnings
import pandas as pd
import sys

warnings.filterwarnings("ignore")

ds = DataSet("/Users/nipunbatra/Downloads/wikienergy-2.h5")

num_appliances_per_home = {}
seen = 0
for building in ds.buildings:
    seen+=1
    sys.stdout.write("\r "+str(seen*100.0/239)+"% done")
    sys.stdout.flush()
    e = ds.buildings[building].elec
    if e is not None:
        num_appliances_per_home[building] = len(e.appliances)
    else:
        print building

df_num_appliances = pd.DataFrame({"num_appliances":num_appliances_per_home})
zero_appliances = df_num_appliances[df_num_appliances["num_appliances"]==0]

buildings_to_ignore = zero_appliances.index.values.tolist()
buildings_to_ignore.extend([199])
print buildings_to_ignore
buildings_to_consider = [l for l in range(1, 239) if l not in buildings_to_ignore]

categories = set()
appliance_types = set()
for appliance in nilmtk.global_meter_group.appliances:
    categories.update(appliance.categories())
    appliance_types.add(appliance.type['type'])

ALL_APPLIANCES = list(appliance_types)
HVAC_APPLIANCES = ['air conditioner', 'electric furnace', 'electric space heater']
LIGHT_APPLIANCES = ['light']
WASHING_APPLIANCES = ['washing machine', 'washer dryer', 'spin dryer']
KITCHEN_APPLIANCES = ['dish washer', 'microwave', 'oven', 'stove']
FRIDGE_APPLIANCES = ['fridge', 'freezer']
WATER_HEATING_APPLIANCES = ['electric hot tub heater', 'electric swimming pool heater', 'electric water heating appliance']
SOCKET_APPLIANCES = ['sockets']
import itertools
ab = itertools.chain(HVAC_APPLIANCES, LIGHT_APPLIANCES, WASHING_APPLIANCES, 
                     KITCHEN_APPLIANCES, FRIDGE_APPLIANCES, WATER_HEATING_APPLIANCES,
                     SOCKET_APPLIANCES)

ALL_LIST = {"hvac": HVAC_APPLIANCES, "light": LIGHT_APPLIANCES, 
            "washing":WASHING_APPLIANCES, "kitchen":KITCHEN_APPLIANCES, 
            "fridge": FRIDGE_APPLIANCES, "water_heating":WATER_HEATING_APPLIANCES,
            "socket":SOCKET_APPLIANCES, 
            "ev": ['electric vehicle'],
            "others":list(set(ALL_APPLIANCES)-set(ab))
            }

out = {}
for building in buildings_to_consider[:]:
    e = ds.buildings[building].elec
    out[building] = {}
    out[building]["total"] = e.mains().total_energy()["active"]
    out[building]["submetered"] = 100*e.proportion_of_energy_submetered()
    for APPLIANCE_LIST_NAME, APPLIANCE_LIST in ALL_LIST.iteritems():
        #out[building][APPLIANCE_LIST_NAME] = {}
        out[building][APPLIANCE_LIST_NAME] = 0
        for appliance in APPLIANCE_LIST:
            m = e.select_using_appliances(type = appliance)
            if len(m.meters)>0:
                out[building][APPLIANCE_LIST_NAME]+= m.total_energy()["active"]
            else:
                out[building][APPLIANCE_LIST_NAME]+= 0

       