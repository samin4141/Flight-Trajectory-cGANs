import pandas as pd
from argparse import ArgumentParser
from time import perf_counter
import multiprocessing as mp
from utm import from_latlon
import numpy as np
import os
from torch.utils.data import Dataset
from torch import save
from random import shuffle

start_time = perf_counter()

parser = ArgumentParser()
parser.add_argument('input')
parser.add_argument('output')
parser.add_argument('-p', '--points', type=int, default=200)
parser.add_argument('-s', '--seconds', type=float, default=2.0)
parser.add_argument('-c', '--cpu', type=int, default=mp.cpu_count() - 1)

args = parser.parse_args()
file_list = os.listdir(args.input)
# Shuffle the list of files to evenly distribute the workload across the cores
shuffle(file_list)
actype_df = pd.read_csv('aircraft_type.csv')
actype_df.columns = ['Manufacturer', 'Model', 'Type Designator', 'Description', 'Engine Type', 'Engine Count', 'WTC']
actype_wtc_dict = dict(zip(actype_df['Type Designator'], actype_df['WTC']))
wtc_to_int_dict = {'M': 0, 'L': 1, 'H': 2, 'L/M': 3, 'J': 4}
north_east = lambda x: from_latlon(x['latitude'],x['longitude'])

class TrajectoryDataset(Dataset):
    def __init__(self, data_arr, label_arr, transform=None, target_transform=None):
        self.data_arr = data_arr
        self.label_arr = label_arr
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.label_arr)

    def __getitem__(self, idx):
        trajectory = self.data_arr[idx]
        label = self.label_arr[idx]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return trajectory, label

def convert_delta(g1):
    g1.groupby(['unique_id','zone', 'band'], as_index=False).nth(0).index[1:]
    # Get index of first row of each zone (except first zone)
    origin_index = g1.groupby(['unique_id','zone', 'band'], as_index=False).nth(0).index[1:]
    # Check if group only has 1 unique zone
    if len(origin_index) == 0:
        g1[['final_delta_e', 'final_delta_n']] = g1[['delta_e', 'delta_n']]
        return g1
    # Calculate inverse of first row of each zone in latitude/longitude
    inverse = g1.loc[origin_index - 1].reset_index()[['latitude','longitude']] - (g1.loc[origin_index].reset_index()[['latitude','longitude']] - g1.loc[origin_index - 1].reset_index()[['latitude','longitude']])
    # Check if inverse is within bounds of longitude/latitude
    if len(inverse[(inverse['latitude'] > 84) | (inverse['latitude'] < -80) | (inverse['longitude'] > 180) | (inverse['longitude'] < -180)]) != 0:
        return
    # Convert inverse to UTM
    inverse_results = inverse.apply(north_east, axis=1, result_type='expand')
    # Check if inverse is within same zone as the last row of the previous zone
    if 0 in (inverse_results[[2,3]].values == g1.loc[origin_index-1][['zone', 'band']].values):
        return
    # Calculate new origin estimate in UTM
    new_origin = g1.loc[origin_index-1][['easting', 'northing']].values - inverse_results[[0,1]].values + g1.loc[origin_index-1][['easting', 'northing']].values
    # Calculate new origin estimate in UTM delta with respect to the first row of the previous zone
    g1.loc[origin_index, ['origin_e', 'origin_n']] = new_origin - g1.loc[origin_index-1][['easting', 'northing']].values + g1.loc[origin_index-1][['delta_e', 'delta_n']].values
    # Calculate final delta in UTM for every row in the group
    g1['delta_cum_n'] = g1['origin_n'].cumsum()
    g1['delta_cum_e'] = g1['origin_e'].cumsum()
    g1['final_delta_n'] = g1['delta_n'] + g1['delta_cum_n']
    g1['final_delta_e'] = g1['delta_e'] + g1['delta_cum_e']
    g1.drop(['delta_cum_n', 'delta_cum_e'], axis=1, inplace=True)

    return g1

def process_data(file_name):
    print(file_name)
    df = pd.read_csv(f'{args.input}/{file_name}')
    initial_length = len(df)

    # actype to wtc to label conversion
    df['wtc'] = df['actype'].map(actype_wtc_dict)
    df.dropna(inplace=True)
    df['label'] = df['wtc'].map(wtc_to_int_dict).astype(int)

    # Filter out flights with less than args.points points
    df['last_position'] = pd.to_datetime(df['last_position'])
    df['filter'] = df.groupby(['callsign', 'icao24'])['last_position'].diff() > pd.Timedelta(args.seconds,'s')
    df['count'] = df.groupby(['callsign', 'icao24'])['filter'].transform(pd.Series.cumsum).astype(str)
    df['subgroup_id'] = (df.groupby(['callsign', 'icao24', 'count']).cumcount() // 200).astype(str)
    df['long'] = df.groupby(['callsign', 'icao24', 'count', 'subgroup_id'])['longitude'].transform('first').astype(str).str[-5:]
    df['unique_id'] = df['callsign'] + '_' + df['icao24'] + '_' + df['count'] + '_' + df['long'] + '_' + df['subgroup_id']
    df = df.groupby('unique_id').filter(lambda group: len(group) == args.points)

    if df.empty:
        return initial_length, None

    # UTM conversion
    df[['easting', 'northing', 'zone', 'band']] = df[['latitude', 'longitude']].apply(north_east, axis=1, result_type='expand')
    df[['origin_e', 'origin_n']] = 0

    # Uncomment to use altitude and geoaltitude and their deltas
    df['origin_a'] = df.groupby('unique_id')['altitude'].transform('first')
    df['delta_a'] = df['altitude'] - df['origin_a']
    # df['origin_ga'] = df.groupby('unique_id')['geoaltitude'].transform('first')
    # df['delta_ga'] = df['geoaltitude'] - df['origin_ga']

    # Calculate delta in UTM with zone jump support
    df[['delta_e', 'delta_n']] = df[['easting', 'northing']] - df.groupby(['unique_id','zone','band'])[['easting', 'northing']].transform('first')
    df = df.groupby(['unique_id'], group_keys=False).apply(lambda x: convert_delta(x)).drop(['origin_e', 'origin_n', 'delta_e', 'delta_n'], axis=1)

    # Add delta_ga, origin_a, origin_ga if using altitude and geoaltitude
    # Always leave label and unique_id as last columns like below
    df = df[['final_delta_n', 'final_delta_e', 'delta_a', 'label', 'unique_id']]
    
    return initial_length, df

if __name__ == '__main__':
    pool = mp.Pool(processes = args.cpu)
    results = pool.map(process_data, file_list)
    pool.close()
    pool.join()

    df_list = [i[1] for i in results if i is not None]
    combined_initial_length = sum([i[0] for i in results if i is not None])

    df = pd.concat(df_list)
    df.to_csv('output.csv', index=False)
    arr = np.array(df.groupby(['unique_id'], sort=False).apply(lambda x: np.array(x) if len(x)==args.points else None).dropna().tolist())[:, :, :-1]
    label_arr = arr[:, :, -1][:, 0].astype(int)
    arr = arr[:, :, :-1].astype(np.float32)

    save(TrajectoryDataset(arr, label_arr), args.output)

    print()
    print(f'Data Retention Rate: {len(df)/combined_initial_length:.2f}')
    print(f'Time Taken: {perf_counter() - start_time}')