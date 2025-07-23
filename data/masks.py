import math

import numpy as np
import torch
import torch.utils.data


class MaskCollator:
  def __init__(self, patch_size, min_block_size, min_keep_ratio, max_keep_ratio):
    self.patch_size = patch_size
    self.min_block_size = min_block_size
    self.min_keep_ratio = min_keep_ratio
    self.max_keep_ratio = max_keep_ratio

  def __call__(self, batch):
    batch = torch.utils.data.default_collate(batch)
    batch_size, num_channels, channel_size = batch.size()
    assert channel_size % self.patch_size == 0
    num_patches = channel_size // self.patch_size
    keep_ratio = np.random.uniform(self.min_keep_ratio, self.max_keep_ratio)
    num_keep = math.ceil(keep_ratio * num_patches)
    mask_encoder, mask_predictor = [], []
    for _ in range(batch_size):
      mask = self.sample_mask(num_keep, num_patches)
      mask_encoder.append(mask.nonzero().squeeze())  # patches to keep
      mask_predictor.append((1 - mask).nonzero().squeeze())  # patches to mask
    mask_encoder = torch.utils.data.default_collate(mask_encoder)
    mask_predictor = torch.utils.data.default_collate(mask_predictor)
    return batch, mask_encoder, mask_predictor

  def sample_mask(self, num_keep, num_patches):  # number of patches to keep (i.e., to not mask)
    # intervals that represent unmasked patches in the mask
    patch_intervals = [(0, num_patches)]
    num_mask = num_patches - num_keep
    total_mask_size = 0  # total number of all masked patches
    while total_mask_size < num_mask:
      interval_sizes = np.diff(patch_intervals).flatten()
      # select a random interval for masking
      index = np.random.choice(len(patch_intervals), p=interval_sizes / interval_sizes.sum())
      start, end = patch_intervals.pop(index)
      interval_size = end - start
      # select a number of consecutive patches to mask, i.e., create a block
      max_block_size = num_mask - total_mask_size
      if max_block_size >= self.min_block_size:
        block_size = np.random.randint(self.min_block_size, max_block_size + 1)
      else:
        block_size = max_block_size
      if interval_size <= block_size:
        # mask entire interval because it is so small, i.e., attach this block to another block
        total_mask_size += interval_size
      else:
        if max_block_size >= self.min_block_size:
          split = np.random.randint(start, end - block_size + 1)  # randomly position this block
        else:
          # this remaining block is too small to be on its own, so attach it to another block
          attach_choices = []
          if start > 0:
            attach_choices.append(start)
          if end < num_patches:
            attach_choices.append(end - block_size)
          split = np.random.choice(attach_choices)
        # split the interval and make place for this new block
        patch_intervals.append((start, split))
        patch_intervals.append((split + block_size, end))
        total_mask_size += block_size
    total_remaining_patches = np.diff(patch_intervals).sum()
    assert total_mask_size + total_remaining_patches == num_patches
    # create the binary mask from blocks and remaining patch intervals
    mask = torch.zeros(num_patches)
    for start, end in patch_intervals:
      mask[start:end] = 1.
    return mask


if __name__ == '__main__':  # visualize masks
  import matplotlib.patches as patches
  import matplotlib.pyplot as plt

  def draw_mask(mask, gap):
    N = len(mask)
    fig, ax = plt.subplots()
    fig.set_size_inches(N + (N - 1) * gap, 1)
    for i, unmasked in enumerate(mask):
      x = i * (1 + gap)  # patch position
      color = 'white' if unmasked else 'black'
      patch = patches.Rectangle((x, 0), 1, 1, edgecolor='black', linewidth=1, facecolor=color)
      ax.add_patch(patch)
    ax.set_aspect('equal')
    ax.set_xlim(0, N + (N - 1) * gap)
    ax.set_ylim(0, 1)
    ax.axis('off')
    plt.show()
  num_patches = 36
  collate = MaskCollator(
    patch_size=1,
    min_block_size=3,
    min_keep_ratio=0.2,
    max_keep_ratio=0.5)
  for _ in range(10):
    keep_ratio = np.random.uniform(collate.min_keep_ratio, collate.max_keep_ratio)
    num_keep = math.ceil(keep_ratio * num_patches)
    mask = collate.sample_mask(num_keep, num_patches)
    draw_mask(mask, gap=0.1)
