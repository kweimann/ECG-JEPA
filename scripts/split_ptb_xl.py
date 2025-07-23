import argparse
from os import path

import numpy as np
import pandas as pd

from data import utils as datautils

parser = argparse.ArgumentParser()
parser.add_argument('--data-dir', required=True, help='path to data directory')
parser.add_argument('--folds', required=True, nargs='+', type=int, help='folds to use')
parser.add_argument('--dump', help='path to dump file (.npy) with raw ECG signals')
args = parser.parse_args()

dump_file = args.dump or f'{args.data_dir}.npy'
if not path.isfile(dump_file):
  raise ValueError(f'Failed to find .npy data file. Attempted location: {dump_file}. '
                   f'Use `--dump` to specify location.')

record_list = pd.read_csv(path.join(args.data_dir, 'ptbxl_database.csv'))
record_list = record_list[record_list.strat_fold.isin(args.folds)]

data = datautils.load_data_dump(dump_file)
data = data[record_list.index]

filename, _ = path.splitext(path.basename(dump_file))

np.save(f'{filename}-split.npy', data)
