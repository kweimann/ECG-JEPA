import multiprocessing as mp

import numpy as np
import wfdb
from torch.utils.data import Dataset, IterableDataset
from tqdm import tqdm


class TensorDataset(Dataset):
  def __init__(self, data, labels=None, transform=None):
    self.data = data
    self.labels = labels
    self.transform = transform

  def __getitem__(self, index):
    x = self.data[index]
    if self.transform is not None:
      if callable(self.transform):
        x = self.transform(x)
      else:
        for transform in self.transform:
          x = transform(x)
    if self.labels is not None:
      y = self.labels[index]
      return x, y
    else:
      return x

  def __len__(self):
    return len(self.data)


class VariableTensorDataset(Dataset):
  # NOTE: this class could inherit from the TensorDataset above, BUT then the dataloader becomes very slow
  def __init__(self, data, starts, sizes, labels=None, transform=None):
    assert len(starts) == len(sizes)
    self.data = data
    self.starts = starts
    self.sizes = sizes
    self.labels = labels
    self.transform = transform

  def __getitem__(self, index):
    start = self.starts[index]
    size = self.sizes[index]
    x = self.data[start:start + size]
    if self.transform is not None:
      if callable(self.transform):
        x = self.transform(x)
      else:
        for transform in self.transform:
          x = transform(x)
    if self.labels is not None:
      y = self.labels[index]
      return x, y
    else:
      return x

  def __len__(self):
    return len(self.starts)


class DatasetRouter(IterableDataset):
  """Samples records from datasets according to a probability distribution."""
  def __init__(self, datasets_with_weights):
    self.datasets, self.weights = zip(*datasets_with_weights)
    assert sum(self.weights) == 1, 'weights must sum up to 1'
    assert all(isinstance(dataset, Dataset) for dataset in self.datasets)
    self.weights = np.array(self.weights)

  def __next__(self):
    dataset_index = np.random.choice(len(self.datasets), p=self.weights)
    dataset = self.datasets[dataset_index]
    record_index = np.random.randint(len(dataset))
    record = dataset[record_index]
    return record

  def __iter__(self):
    return self


def get_channel_order(source_channels, target_channels):
  """Returns order for the source channels to match the target channels."""
  channel_index = {c.casefold(): i for i, c in enumerate(source_channels)}
  channel_order = [channel_index[c.casefold()] for c in target_channels]
  return channel_order


def load_raw_data(record_names, min_channel_size=None, dtype=np.float16, verbose=False):
  # if min_channel_size is provided, we filter records based on channel size,
  #  so the data shape cannot be computed before scanning all records
  # furthermore, we assume that all recordings have the same channel size
  data = None if min_channel_size is None else []
  for i, record_name in enumerate(tqdm(record_names, disable=not verbose)):
    x = wfdb.rdrecord(record_name)
    x = x.p_signal
    if data is None:
      num_records = len(record_names)
      data = np.empty((num_records, *x.shape), dtype=dtype)
    if min_channel_size is not None:
      channel_size, num_channels = x.shape
      if channel_size < min_channel_size:
        continue
      x = x.astype(dtype)
      data.append(x)
    else:
      data[i] = x
  if min_channel_size is not None:
    data = np.array(data)
  return data


def load_raw_variable_data(record_names, dtype=np.float16, verbose=False):
  data = []
  for record_name in tqdm(record_names, disable=not verbose):
    x = wfdb.rdrecord(record_name)
    x = x.p_signal
    x = x.astype(dtype)
    data.append(x)
  sizes = np.array([len(x) for x in data])
  data = np.concatenate(data)
  return data, sizes


def load_data_dump(dump_file, transform=None, processes=None, chunk_size=32):
  """Loads data into memory and optionally preprocesses it."""
  if transform is None:
    return np.load(dump_file)
  original_data = np.load(dump_file, mmap_mode='r')
  num_records = len(original_data)
  data = None
  with mp.Pool(
      processes=processes or mp.cpu_count(),
      initializer=_init_worker,
      initargs=(dump_file, transform, chunk_size)
  ) as pool:
    chunks = range(0, num_records, chunk_size)
    for index, chunk in zip(chunks, pool.imap(_preprocess, chunks)):
      if data is None:
        record_shape = chunk[0].shape
        dtype = chunk[0].dtype
        data = np.empty((num_records, *record_shape), dtype=dtype)
      data[index:index + len(chunk)] = chunk
  return data


def load_variable_data_dump(dump_file, transform=None, processes=None, chunk_size=32):
  data_archive = np.load(dump_file)
  original_data, original_sizes = data_archive['data'], data_archive['sizes']
  original_starts = np.concatenate([[0], np.cumsum(original_sizes[:-1])])
  if transform is None:
    data = [original_data[start:start + size]
            for start, size in zip(original_starts, original_sizes)]
  else:
    def iter_original_data():
      for start, size in zip(original_starts, original_sizes):
        yield original_data[start:start + size]
    data = []
    with mp.Pool(processes=processes or mp.cpu_count()) as pool:
      for x in pool.imap(transform, iter_original_data(), chunksize=chunk_size):
        data.append(x)
  return data


def _init_worker(file, transform, chunk_size):
  global _data, _transform, _chunk_size
  _data = np.load(file, mmap_mode='r')
  _transform = transform
  _chunk_size = chunk_size


def _preprocess(index):
  chunk = _data[index:index + _chunk_size].copy()
  chunk = [_transform(x_i) for x_i in chunk]
  return chunk
