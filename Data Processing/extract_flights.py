import pandas as pd
from argparse import ArgumentParser
from time import perf_counter

start_time = perf_counter()

parser = ArgumentParser()
parser.add_argument('input')
parser.add_argument('output')
parser.add_argument('-p', '--points', type=int, default=200)
parser.add_argument('-s', '--seconds', type=float, default=2.0)

args = parser.parse_args()

df = pd.read_csv(args.input)
initial_length = len(df)
df['last_position'] = pd.to_datetime(df['last_position'])
df['filter'] = df.groupby(['callsign', 'icao24'])['last_position'].diff() > pd.Timedelta(args.seconds,'s')
df['count'] = df.groupby(['callsign', 'icao24'])['filter'].transform(pd.Series.cumsum).astype(str)
df['long'] = df.groupby(['callsign', 'icao24', 'count'])['longitude'].transform('first').astype(str).str[-5:]
df['unique_id'] = df['callsign'] + '_' + df['icao24'] + '_' + df['count'] + '_' + df['long']
out = df.groupby(['unique_id']).filter(lambda group: len(group) > args.points)
out.drop(['filter', 'count', 'long'], axis=1, inplace=True)

print(f'Data Retention Rate: {len(out)/initial_length:.2f}')

out.to_csv(args.output)

print(f'Time Taken: {perf_counter() - start_time}')