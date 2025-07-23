import argparse
from os import path

import numpy as np

from data.datasets import *
from data.utils import load_raw_data, load_raw_variable_data

parser = argparse.ArgumentParser()
parser.add_argument('--data-dir', required=True, help='path to data directory')
parser.add_argument('--dataset', choices=list(DATASETS), help='dataset type')
parser.add_argument('--verbose', action='store_true', help='verbose mode')
args = parser.parse_args()

if args.dataset is None:
  dir_name = path.basename(args.data_dir)
  if dir_name == 'chapman_shaoxing':
    args.dataset = 'chapman-shaoxing'
  elif dir_name == 'cpsc_2018':
    args.dataset = 'cpsc'
  elif dir_name == 'cpsc_2018_extra':
    args.dataset = 'cpsc-extra'
  elif dir_name == 'georgia':
    args.dataset = 'georgia'
  elif dir_name == 'ningbo':
    args.dataset = 'ningbo'
  elif dir_name == 'ptb':
    args.dataset = 'ptb'
  elif dir_name == 'st_petersburg_incart':
    args.dataset = 'st-petersburg'
  elif dir_name.startswith('code-15'):
    args.dataset = 'code-15'
  elif dir_name.startswith('mimic-iv-ecg'):
    args.dataset = 'mimic-iv-ecg'
  elif dir_name.startswith('ptb-xl'):
    args.dataset = 'ptb-xl'
  else:
    raise ValueError(f'Failed to infer dataset type from data directory {args.data_dir}. '
                     f'Use `--dataset` to provide the dataset type.')
  print(f'Inferred dataset type is {args.dataset}')

print(f'Loading data from {args.data_dir}')

data, sizes = None, None

if args.dataset == 'chapman-shaoxing':
  record_names = ChapmanShaoxing.find_records(args.data_dir)
  data = load_raw_data(record_names, verbose=args.verbose)
elif args.dataset == 'cpsc':
  record_names = CPSC2018.find_records(args.data_dir)
  data, sizes = load_raw_variable_data(record_names, verbose=args.verbose)
elif args.dataset == 'cpsc-extra':
  record_names = CPSC2018Extra.find_records(args.data_dir)
  data, sizes = load_raw_variable_data(record_names, verbose=args.verbose)
elif args.dataset == 'georgia':
  record_names = Georgia.find_records(args.data_dir)
  channel_size = Georgia.record_duration * Georgia.sampling_frequency
  data = load_raw_data(record_names, min_channel_size=channel_size, verbose=args.verbose)
elif args.dataset == 'ningbo':
  record_names = ChapmanShaoxing.find_records(args.data_dir)
  data = load_raw_data(record_names, verbose=args.verbose)
elif args.dataset == 'ptb':
  record_names = PTB.find_records(args.data_dir)
  data, sizes = load_raw_variable_data(record_names, verbose=args.verbose)
elif args.dataset == 'st-petersburg':
  record_names = StPetersburg.find_records(args.data_dir)
  data = load_raw_data(record_names, verbose=args.verbose)
elif args.dataset == 'code-15':
  dataset = CODE15(args.data_dir)
  data = dataset.load_raw_data(skip_variable=True, verbose=args.verbose)
elif args.dataset == 'mimic-iv-ecg':
  record_names = MIMIC_IV_ECG.find_records(args.data_dir)
  data = load_raw_data(record_names, verbose=args.verbose)
elif args.dataset == 'ptb-xl':
  record_names = PTB_XL.find_records(args.data_dir)
  data = load_raw_data(record_names, verbose=args.verbose)
else:
  raise ValueError(f'Unknown dataset type: {args.dataset}')

if sizes is None:  # we are not dealing with variable data
  out_file = f'{args.data_dir}.npy'
  print(f'Saving dataset to {out_file}')
  np.save(out_file, data)
else:
  out_file = f'{args.data_dir}.npz'
  print(f'Saving dataset to {out_file}')
  np.savez(out_file, data=data, sizes=sizes)
