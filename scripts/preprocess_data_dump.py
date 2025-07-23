import argparse
import functools
from os import path

import numpy as np

from data import transforms, utils as datautils
from data.datasets import DATASETS
from utils.monitoring import get_cpu_count

parser = argparse.ArgumentParser()
parser.add_argument('--data', required=True, help='path to the dump file')
parser.add_argument('--interpolate-nans', action='store_true', help='interpolate NaNs')
parser.add_argument('--remove-baseline-wander', action='store_true', help='remove baseline wander')
parser.add_argument('--dataset', choices=list(DATASETS), help='dataset type')
args = parser.parse_args()

name, ext = path.splitext(args.data)

if args.dataset is None:
  basename = path.basename(name)
  if basename == 'chapman_shaoxing':
    args.dataset = 'chapman-shaoxing'
  elif basename == 'cpsc_2018':
    args.dataset = 'cpsc'
  elif basename == 'cpsc_2018-extra':
    args.dataset = 'cpsc-extra'
  elif basename == 'georgia':
    args.dataset = 'georgia'
  elif basename == 'ningbo':
    args.dataset = 'ningbo'
  elif basename == 'ptb':
    args.dataset = 'ptb'
  elif basename == 'st_petersburg_incart':
    args.dataset = 'st-petersburg'
  elif basename.startswith('code-15'):
    args.dataset = 'code-15'
  elif basename.startswith('mimic-iv-ecg'):
    args.dataset = 'mimic-iv-ecg'
  elif basename.startswith('ptb-xl'):
    args.dataset = 'ptb-xl'
  else:
    raise ValueError(f'Failed to infer dataset type from data directory {args.data_dir}')
  print(f'Inferred dataset type is {args.dataset}')

transform_fns = []

if args.interpolate_nans:
  print('Transforms: NaN interpolation')
  transform_fns.append(transforms.interpolate_NaNs_)

if args.remove_baseline_wander:
  print('Transforms: baseline wander removal')
  dataset = DATASETS[args.dataset]
  fs = dataset.sampling_frequency
  highpass_filter = functools.partial(transforms.highpass_filter, fs=fs)
  transform_fns.append(highpass_filter)

if not transform_fns:
  print('No preprocessing steps were chosen')
else:
  def chain(fns):
    def chained(x):
      for fn in fns:
        x = fn(x)
      return x
    return chained
  transform = chain(transform_fns)
  if ext == '.npy':
    data = datautils.load_data_dump(args.data, transform=transform, processes=get_cpu_count())
    np.save(f'{name}-preprocessed{ext}', data)
  elif ext == '.npz':  # variable data
    data = datautils.load_variable_data_dump(args.data, transform=transform, processes=get_cpu_count())
    sizes = np.array([len(x) for x in data])
    data = np.concatenate(data)
    np.savez(f'{name}-preprocessed{ext}', data=data, sizes=sizes)
  else:
    raise ValueError(f'Unsupported dataset: {args.data}')
