from collections import OrderedDict

import torch
from torch import nn

import configs
from models import VisionTransformer
from models.modules import AttentivePooler


class ViTClassifier(nn.Module):
  def __init__(self, encoder: VisionTransformer, config: configs.eval.Config, use_sdp_kernel=True):
    super().__init__()
    self.config = config
    self.encoder = encoder
    if config.attn_pooling:
      self.attn_pool = AttentivePooler(
        dim=encoder.config.dim,
        num_heads=encoder.config.num_heads,
        num_queries=1,
        mlp_ratio=encoder.config.mlp_ratio,
        qkv_bias=encoder.config.qkv_bias,
        bias=encoder.config.bias,
        proj_dim=config.num_classes,
        eps=encoder.config.norm_eps,
        use_sdp_kernel=use_sdp_kernel)
    else:
      assert not config.use_register or (encoder.config.num_registers > 0 and encoder.keep_registers)
      self.fc = nn.Linear(encoder.config.dim, config.num_classes, bias=config.bias)

    for name, module in self.named_modules(memo=set(encoder.modules())):
      if isinstance(module, nn.Linear):
        nn.init.trunc_normal_(module.weight, mean=0., std=0.02)
        if module.bias is not None:
          nn.init.zeros_(module.bias)
      elif isinstance(module, nn.LayerNorm):
        nn.init.ones_(module.weight)
        if module.bias is not None:
          nn.init.zeros_(module.bias)

    if self.config.frozen:
      for parameter in self.encoder.parameters():
        parameter.requires_grad = False

  def encode(self, x):
    if self.config.frozen:
      with torch.no_grad():
        x = self.encoder(x)
    else:
      x = self.encoder(x)
    return x

  def forward(self, x, encoded=False):
    if not encoded:
      x = self.encode(x)
    if self.config.attn_pooling:
      x = self.attn_pool(x).squeeze(dim=1)
    else:
      if self.config.use_register:
        x = x[:, 0]
      else:
        x = x.mean(dim=1)  # global average pooling
      x = self.fc(x)
    return x

  def get_optimizer(self, fused=False):
    decay_modules = (nn.Linear, nn.Conv1d)
    decay = set()
    for module_name, module in self.named_modules():
      for param_name, param in module.named_parameters():
        if isinstance(module, decay_modules) and param_name.endswith('weight') and param.requires_grad:
          param_name = f'{module_name}.{param_name}' if module_name else param_name
          decay.add(param_name)

    decay_params, non_decay_params = OrderedDict(), OrderedDict()
    for name, param in self.named_parameters():
      if param.requires_grad:
        if name in decay:
          decay_params[name] = param
        else:
          non_decay_params[name] = param

    param_groups = [
      {'params': list(decay_params.values()),
       'weight_decay': self.config.weight_decay,
       'use_weight_decay': True},
      {'params': list(non_decay_params.values()),
       'weight_decay': 0.,
       'use_weight_decay': False}
    ]

    optimizer = torch.optim.AdamW(
      param_groups,
      lr=self.config.learning_rate,
      betas=self.config.opt_betas,
      weight_decay=0.,
      fused=fused)

    return optimizer
