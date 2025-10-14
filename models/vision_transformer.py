import torch
from torch import nn

import configs
from models.modules import PatchEmbedding, Block
from models.utils import get_1d_pos_embed, apply_mask


class VisionTransformer(nn.Module):
  def __init__(self, config: configs.pretrain.Config, keep_registers=False, use_sdp_kernel=True):
    super().__init__()
    self.config = config
    self.keep_registers = keep_registers
    assert config.channel_size % config.patch_size == 0
    num_patches = config.channel_size // config.patch_size
    self.patch_embed = PatchEmbedding(
      dim=config.dim,
      in_channels=config.num_channels,
      patch_size=config.patch_size,
      bias=config.bias)
    self.register_buffer(
      'pos_embed',
      get_1d_pos_embed(
        dim=config.dim,
        num_patches=num_patches),
      persistent=False)
    if config.num_registers > 0:
      self.registers = nn.Parameter(torch.empty(1, config.num_registers, config.dim))
      nn.init.trunc_normal_(self.registers, mean=0., std=0.02)
    self.blocks = nn.ModuleList([
      Block(
        dim=config.dim,
        num_heads=config.num_heads,
        mlp_ratio=config.mlp_ratio,
        qkv_bias=config.qkv_bias,
        bias=config.bias,
        dropout=config.dropout,
        attn_dropout=config.attn_dropout,
        eps=config.norm_eps,
        layer_scale_eps=config.layer_scale_eps,
        use_sdp_kernel=use_sdp_kernel)
      for _ in range(config.depth)
    ])
    self.norm = nn.LayerNorm(config.dim, eps=config.norm_eps, bias=config.bias)

    for name, module in self.named_modules():
      if isinstance(module, (nn.Linear, nn.Conv1d)):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
          nn.init.zeros_(module.bias)
      elif isinstance(module, nn.LayerNorm):
        nn.init.ones_(module.weight)
        if module.bias is not None:
          nn.init.zeros_(module.bias)

  def forward(self, x, mask=None):
    x = self.patch_embed(x)
    B, N, D = x.size()
    x = x + self.pos_embed[:, :N]
    if mask is not None:
      x = apply_mask(x, mask)
    if self.config.num_registers > 0:
      registers = self.registers.repeat(B, 1, 1)
      x = torch.cat([registers, x], dim=1)
    for block in self.blocks:
      x = block(x)
    x = self.norm(x)
    if not self.keep_registers and self.config.num_registers > 0:
      x = x[:, self.config.num_registers:]
    return x
