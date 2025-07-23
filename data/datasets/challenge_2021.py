import glob
from os import path


class Challenge2021:  # collection of some datasets from PhysioNet Challenge 2021 (version 1.0.3)
  channels = (
    'I', 'II', 'III', 'aVR', 'aVL', 'aVF',
    'V1', 'V2', 'V3', 'V4', 'V5', 'V6'
  )

  @staticmethod
  def find_records(data_dir):
    header_files = glob.glob(path.join(data_dir, '**', '*.hea'), recursive=True)
    record_names = [path.splitext(header_file)[0] for header_file in header_files]
    return record_names


class ChapmanShaoxing(Challenge2021):
  sampling_frequency = 500
  record_duration = 10
  mean = (
    0.002, 0.005, 0.003, -0.002, -0.002, 0.003,
    -0.000, 0.003, 0.004, 0.004, 0.003, -0.001
  )
  std = (
    0.140, 0.171, 0.145, 0.138, 0.114, 0.143,
    0.218, 0.355, 0.366, 0.365, 0.332, 0.299
  )
  # shape: (10247, 5000, 12)


class CPSC2018(Challenge2021):
  sampling_frequency = 500
  mean = (
    0.000, 0.000, 0.000, 0.000, 0.000, 0.000,
    0.000, 0.000, 0.000, 0.000, 0.000, 0.000,
  )
  std = (
    0.214, 0.236, 0.221, 0.209, 0.196, 0.214,
    0.336, 0.417, 0.449, 0.483, 0.523, 0.561
  )
  # shape: (54837488, 12)
  # records: 6877


class CPSC2018Extra(Challenge2021):
  sampling_frequency = 500
  mean = (
    0.000, 0.000, 0.000, 0.000, 0.000, 0.000,
    0.000, 0.000, 0.000, 0.000, 0.000, 0.000,
  )
  std = (
    0.182, 0.183, 0.184, 0.163, 0.164, 0.165,
    0.288, 0.370, 0.371, 0.380, 0.375, 0.392
  )
  # shape: (27466844, 12)
  # records: 3453


class Georgia(Challenge2021):
  sampling_frequency = 500
  record_duration = 10  # we filter all records below 10 seconds
  mean = (
    -0.003, -0.002, 0.001, 0.003, -0.002, -0.001,
    -0.004, -0.004, -0.003, -0.003, -0.003, -0.003
  )
  std = (
    0.157, 0.178, 0.164, 0.147, 0.134, 0.153,
    0.213, 0.285, 0.272, 0.245, 0.224, 0.209
  )
  # shape: (10292, 5000, 12)


class Ningbo(Challenge2021):
  sampling_frequency = 500
  record_duration = 10
  mean = (
    0.002, 0.004, 0.002, -0.001, -0.001, 0.002,
    -0.000, 0.003, 0.003, 0.004, 0.003, -0.000
  )
  std = (
    0.145, 0.179, 0.159, 0.142, 0.123, 0.153,
    0.228, 0.368, 0.377, 0.394, 0.405, 0.444
  )
  # shape: (34905, 5000, 12)


class PTB(Challenge2021):
  sampling_frequency = 1000
  mean = (
    0.000, 0.000, 0.000, 0.000, 0.000, 0.000,
    0.000, 0.000, 0.000, 0.000, 0.000, 0.000,
  )
  std = (
    0.642, 0.975, 0.953, 0.674, 0.651, 0.909,
    0.643, 0.658, 0.668, 0.688, 0.648, 0.622
  )
  # shape: (57156947, 12)
  # records: 516


class StPetersburg(Challenge2021):
  sampling_frequency = 257
  record_duration = 1800
  mean = (
    -0.018, -0.041, 0.004, -0.247, -0.337, -0.269,
    0.622, 0.563, 0.011, -1.321, -0.379, 0.028
  )
  std = (
    2.576, 3.293, 3.464, 2.418, 2.396, 3.142,
    2.195, 2.454, 4.468, 5.193, 3.089, 3.837
  )
  # shape: (74, 462600, 12)
