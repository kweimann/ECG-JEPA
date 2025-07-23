import numpy as np
from scipy import signal


def interpolate_NaNs_(x):  # x: (channel_size, num_channels)
  # this transformation is in-place
  nan_mask = np.isnan(x)
  for index, contains_nans in enumerate(nan_mask.any(axis=0)):
    if contains_nans:
      mask = nan_mask[:, index]
      x[mask, index] = np.interp(
        np.flatnonzero(mask),
        np.flatnonzero(~mask),
        x[~mask, index])
  return x


def normalize_(x, mean_std=None, eps=0):  # x: (channel_size, num_channels)
  # this transformation is in-place
  if mean_std is None:
    mean = x.mean(axis=0, keepdims=True)
    std = x.std(axis=0, keepdims=True)
  else:
    mean, std = mean_std
  x -= mean
  x /= std + eps
  return x


def highpass_filter(x, fs):  # x: (channel_size, num_channels)
  dtype = x.dtype
  [b, a] = signal.butter(4, 0.5, btype='highpass', fs=fs)
  x = signal.filtfilt(b, a, x, axis=0)
  x = x.astype(dtype)
  return x


def resample(x, channel_size):  # x: (channel_size, num_channels)
  dtype = x.dtype
  x = signal.resample(x, channel_size, axis=0)
  x = x.astype(dtype)
  return x


def random_crop(x, size):  # x: (channel_size, num_channels)
  start = np.random.randint(len(x) - size + 1)
  x = x[start:start + size]
  return x
