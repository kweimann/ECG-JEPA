import argparse

import numpy as np
from os import path
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--data', required=True, help='path to the dump file (.npy)')
parser.add_argument('--verbose', action='store_true', help='verbose mode')
args = parser.parse_args()

_, ext = path.splitext(args.data)
if ext == '.npz':
  # we are dealing with variable data; for simplicity we load it all into memory
  #  and compute mean and std the standard way
  data_archive = np.load(args.data)
  data = data_archive['data']
  mean = np.nanmean(data, axis=0, dtype=np.float64)
  std = np.nanstd(data, axis=0, dtype=np.float64)
elif ext == '.npy':
  data = np.load(args.data, mmap_mode='r')
  num_records, channel_size, num_channels = data.shape

  sum_x = np.zeros((num_channels,), dtype=np.float64)
  sum_x2 = np.zeros((num_channels,), dtype=np.float64)
  n = np.zeros((num_channels,), dtype=np.int64)

  for x in tqdm(data, disable=not args.verbose):
    x = x.astype(np.float32)
    sum_x += np.nansum(x, axis=0)
    sum_x2 += np.nansum(x * x, axis=0)
    n += np.sum(~np.isnan(x), axis=0)

  mean = sum_x / n
  std = np.sqrt((sum_x2 - (sum_x * sum_x) / n) / (n - 1))
else:
  raise ValueError(f'Unsupported dataset: {args.data}')

np.set_printoptions(precision=3)
print('mean', mean)
print('std', std)
