from os import path

import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm


class CODE15:
  sampling_frequency = 400
  record_duration = 10.24  # we filter all records below 10.24 seconds
  channels = (
    'I', 'II', 'III', 'AVR', 'AVL', 'AVF',
    'V1', 'V2', 'V3', 'V4', 'V5', 'V6'
  )
  mean = (
    0.053, -0.189, -0.242, 0.068, 0.148, -0.215,
    -0.255, -0.266, -0.226, -0.233, -0.219, -0.288
  )
  std = (
    1.528, 2.657, 2.645, 1.718, 1.704, 2.538,
    2.060, 2.116, 2.055, 1.968, 2.053, 2.315
  )
  # shape: (128033, 4096, 12)

  def __init__(self, data_dir):
    self.record_list = pd.read_csv(path.join(data_dir, 'exams.csv'), index_col='exam_id')
    h5_filenames = self.record_list.trace_file.unique()
    self.h5_files = {name: h5py.File(path.join(data_dir, name), 'r') for name in h5_filenames}
    self.exam_id_to_index = {exam_id: i for h5_file in self.h5_files.values()
                             for i, exam_id in enumerate(h5_file['exam_id'])}

  def stream_raw_data(self):
    for exam_id, trace_file in self.record_list.trace_file.items():
      index = self.exam_id_to_index[exam_id]
      dataset = self.h5_files[trace_file]['tracings']
      x = dataset[index]
      yield x

  def load_raw_data(self, dtype=np.float16, skip_variable=False, verbose=False):
    complete_data = []
    variable_data = []
    num_channels = len(CODE15.channels)
    max_channel_size = int(CODE15.record_duration * CODE15.sampling_frequency)
    for x in tqdm(self.stream_raw_data(), total=len(self.record_list), disable=not verbose):
      slices = []
      for channel in range(num_channels):
        start, end = trim_zeros(x[:, channel])
        slices.append((start, end, end - start))
      starts, ends, channel_sizes = zip(*slices)
      start, end = min(starts), max(ends)
      channel_size = end - start
      if len(np.unique(channel_sizes)) == 1 and channel_size == max_channel_size:
        x = x.astype(dtype)
        complete_data.append(x)
      elif not skip_variable:
        x_ = np.zeros((channel_size, num_channels), dtype=dtype)
        for k, (slice_start, slice_end, _) in enumerate(slices):
          x_[slice_start - start:slice_end - start, k] = x[slice_start:slice_end, k]
        variable_data.append(x_)
    complete_data = np.array(complete_data)
    if skip_variable:
      return complete_data
    else:
      variable_sizes = np.array([len(x) for x in variable_data])
      variable_data = np.concatenate(variable_data)
      return complete_data, (variable_data, variable_sizes)


def trim_zeros(x):
  indices, = np.nonzero(x)
  if len(indices) == 0:
    start = end = -1
  else:
    start = indices[0]
    end = indices[-1] + 1
  return start, end
