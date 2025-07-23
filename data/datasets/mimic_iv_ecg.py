from os import path

import pandas as pd


class MIMIC_IV_ECG:
  sampling_frequency = 500
  record_duration = 10
  channels = (
    'I', 'II', 'III', 'aVR', 'aVF', 'aVL',
    'V1', 'V2', 'V3', 'V4', 'V5', 'V6'
  )
  mean = (
    0.023, 0.015, -0.007, -0.019, 0.004, 0.014,
    -0.018, -0.004, 0.002, 0.017, 0.022, 0.022
  )
  std = (
    0.155, 0.158, 0.166, 0.133, 0.143, 0.141,
    0.207, 0.282, 0.284, 0.253, 0.222, 0.197
  )
  # shape: (800035, 5000, 12)

  @staticmethod
  def find_records(data_dir):
    record_list = pd.read_csv(path.join(data_dir, 'record_list.csv'))
    record_names = [path.join(data_dir, filename) for filename in record_list.path.values]
    return record_names
