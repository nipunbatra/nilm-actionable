from __future__ import print_function, division
import os
from os.path import join, isdir, dirname, abspath
from inspect import currentframe, getfile, getsourcefile

import pandas as pd
import yaml
from nilmtk.measurement import LEVEL_NAMES
from nilmtk.datastore import Key
from nilm_metadata import convert_yaml_to_hdf5

script_path = os.path.dirname(os.path.realpath(__file__))

feed_mapping = {
    'use': {},
    'air1': {'type': 'air conditioner'},
    'air2': {'type': 'air conditioner'},
    'air3': {'type': 'air conditioner'},
    'airwindowunit1': {'type': 'air conditioner'},
    'aquarium1': {'type': 'appliance'},
    'bathroom1': {'type': 'sockets', 'room': 'bathroom'},
    'bathroom2': {'type': 'sockets', 'room': 'bathroom'},
    'bedroom1': {'type': 'sockets', 'room': 'bedroom'},
    'bedroom2': {'type': 'sockets', 'room': 'bedroom'},
    'bedroom3': {'type': 'sockets', 'room': 'bedroom'},
    'bedroom4': {'type': 'sockets', 'room': 'bedroom'},
    'bedroom5': {'type': 'sockets', 'room': 'bedroom'},
    'car1': {'type': 'electric vehicle'},
    'clotheswasher1': {'type': 'washing machine'},
    'clotheswasher_dryg1': {'type': 'washer dryer'},
    'diningroom1': {'type': 'sockets', 'room': 'dining room'},
    'diningroom2': {'type': 'sockets', 'room': 'dining room'},
    'dishwasher1': {'type': 'dish washer'},
    'disposal1': {'type': 'waste disposal unit'},
    'drye1': {'type': 'spin dryer'},
    'dryg1': {'type': 'spin dryer'},
    'freezer1': {'type': 'freezer'},
    'furnace1': {'type': 'electric furnace'},
    'furnace2': {'type': 'electric furnace'},
    'garage1': {'type': 'sockets', 'room': 'dining room'},
    'garage2': {'type': 'sockets', 'room': 'dining room'},
    'gen': {},
    'grid': {},
    'heater1': {'type': 'electric space heater'},
    'housefan1': {'type': 'electric space heater'},
    'icemaker1': {'type': 'appliance'},
    'jacuzzi1': {'type': 'electric hot tub heater'},
    'kitchen1': {'type': 'sockets', 'room': 'kitchen'},
    'kitchen2': {'type': 'sockets', 'room': 'kitchen'},
    'kitchenapp1': {'type': 'sockets', 'room': 'kitchen'},
    'kitchenapp2': {'type': 'sockets', 'room': 'kitchen'},
    'lights_plugs1': {'type': 'light'},
    'lights_plugs2': {'type': 'light'},
    'lights_plugs3': {'type': 'light'},
    'lights_plugs4': {'type': 'light'},
    'lights_plugs5': {'type': 'light'},
    'lights_plugs6': {'type': 'light'},
    'livingroom1': {'type': 'sockets', 'room': 'living room'},
    'livingroom2': {'type': 'sockets', 'room': 'living room'},
    'microwave1': {'type': 'microwave'},
    'office1': {'type': 'sockets', 'room': 'office'},
    'outsidelights_plugs1': {'type': 'sockets', 'room': 'outside'},
    'outsidelights_plugs2': {'type': 'sockets', 'room': 'outside'},
    'oven1': {'type': 'oven'},
    'oven2': {'type': 'oven'},
    'pool1': {'type': 'electric swimming pool heater'},
    'pool2': {'type': 'electric swimming pool heater'},
    'poollight1': {'type': 'light'},
    'poolpump1': {'type': 'electric swimming pool heater'},
    'pump1': {'type': 'appliance'},
    'range1': {'type': 'stove'},
    'refrigerator1': {'type': 'fridge'},
    'refrigerator2': {'type': 'fridge'},
    'security1': {'type': 'security alarm'},
    'shed1': {'type': 'sockets', 'room': 'shed'},
    'sprinkler1': {'type': 'appliance'},
    'unknown1': {'type': 'unknown'},
    'unknown2': {'type': 'unknown'},
    'unknown3': {'type': 'unknown'},
    'unknown4': {'type': 'unknown'},
    'utilityroom1': {'type': 'sockets', 'room': 'utility room'},
    'venthood1': {'type': 'appliance'},
    'waterheater1': {'type': 'electric water heating appliance'},
    'waterheater2': {'type': 'electric water heating appliance'},
    'winecooler1': {'type': 'appliance'},
}


def _dataport_dataframe_to_hdf(df,
                               store,
                               nilmtk_building_id,
                               dataport_building_id):
    local_dataframe = df
    # set timestamp as frame index
    # local_dataframe = local_dataframe.set_index('localminute')

    feeds_dataframe = local_dataframe

    # Column names for dataframe
    column_names = [('power', 'active')]

    # building metadata
    building_metadata = {}
    building_metadata['instance'] = nilmtk_building_id
    building_metadata['original_name'] = int(dataport_building_id)  # use python int
    building_metadata['elec_meters'] = {}
    building_metadata['appliances'] = []

    # initialise dict of instances of each appliance type
    instance_counter = {}

    meter_id = 1
    for column in feeds_dataframe.columns:
        if feeds_dataframe[column].notnull().sum() > 0 and not column in feed_ignore:

            # convert timeseries into dataframe
            feed_dataframe = pd.DataFrame(feeds_dataframe[column])

            # set column names
            feed_dataframe.columns = pd.MultiIndex.from_tuples(column_names)

            # Modify the column labels to reflect the power measurements recorded.
            feed_dataframe.columns.set_names(LEVEL_NAMES, inplace=True)

            key = Key(building=nilmtk_building_id, meter=meter_id)
            print(key)
            print(feed_dataframe.head())
            print(column)

            # store dataframe
            store.put(str(key), feed_dataframe, format='table', append=True)
            store.flush()

            # elec_meter metadata
            if column == 'use':
                meter_metadata = {'device_model': 'eGauge',
                                  'site_meter': True}
            else:
                meter_metadata = {'device_model': 'eGauge',
                                  'submeter_of': 0}
            building_metadata['elec_meters'][meter_id] = meter_metadata

            # appliance metadata
            if column != 'use':
                # original name and meter id
                appliance_metadata = {'original_name': column,
                                      'meters': [meter_id]}
                # appliance type and room if available
                appliance_metadata.update(feed_mapping[column])
                # appliance instance number
                if instance_counter.get(appliance_metadata['type']) == None:
                    instance_counter[appliance_metadata['type']] = 0
                instance_counter[appliance_metadata['type']] += 1
                appliance_metadata['instance'] = instance_counter[appliance_metadata['type']]

                building_metadata['appliances'].append(appliance_metadata)

            meter_id += 1

    # write building yaml to file
    building = 'building{:d}'.format(nilmtk_building_id)
    yaml_full_filename = join(_get_module_directory(), 'metadata', building + '.yaml')
    with open(yaml_full_filename, 'w') as outfile:
        outfile.write(yaml.dump(building_metadata))

    return 0


def _get_module_directory():
    # Taken from http://stackoverflow.com/a/6098238/732596
    path_to_this_file = dirname(getfile(currentframe()))
    if not isdir(path_to_this_file):
        encoding = os.getfilesystemencoding()
        path_to_this_file = dirname(unicode(__file__, encoding))
    if not isdir(path_to_this_file):
        abspath(getsourcefile(lambda _: None))
    if not isdir(path_to_this_file):
        path_to_this_file = os.getcwd()
    assert isdir(path_to_this_file), path_to_this_file + ' is not a directory'
    return path_to_this_file


feed_ignore = ['gen', 'grid']

WEATHER_HVAC_STORE = os.path.join(script_path, '..', '..', '..', 'data/hvac/weather_hvac_2013.h5')

store_total = pd.HDFStore("/Users/nipunbatra/Downloads/wiki-temp.h5")

store_useful = pd.HDFStore(WEATHER_HVAC_STORE)
useful_keys = [k[:-2] for k in store_useful.keys() if "X" in k]

START, STOP = "2013-07-01", "2013-07-31"

store_name = "/Users/nipunbatra/wikienergy-2013.h5"
with pd.HDFStore(store_name, "w") as store_to_write:

    for nilmtk_id, dataid_str in enumerate(useful_keys):

        dataid = int(dataid_str[1:])

        df = store_total[dataid_str][START:STOP]
        if df['air1'].sum()>0:
            print("Writing ", nilmtk_id, dataid)
            _dataport_dataframe_to_hdf(df, store_to_write, nilmtk_id + 1, dataid)
        else:
            print ("Skipping", nilmtk_id, dataid)
    convert_yaml_to_hdf5(join(_get_module_directory(), 'metadata'),
                         store_name)
