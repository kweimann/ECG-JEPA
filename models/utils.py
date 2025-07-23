import math

import torch


def apply_mask(x, mask):  # x: (B, N, D); mask: (B, K) with values in range [0, N)
  mask = mask.unsqueeze(-1).repeat(1, 1, x.size(-1))
  x = torch.gather(x, dim=1, index=mask)
  return x  # x: (B, K, D)


def get_1d_pos_embed(dim, num_patches):
  assert dim % 2 == 0
  position = torch.arange(num_patches).unsqueeze(1)
  div_term = torch.exp(torch.arange(0, dim, 2) * (-math.log(10000.0) / dim))
  pos_embed = torch.zeros(1, num_patches, dim)
  pos_embed[0, :, 0::2] = torch.sin(position * div_term)
  pos_embed[0, :, 1::2] = torch.cos(position * div_term)
  return pos_embed
