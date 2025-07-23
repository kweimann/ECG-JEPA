import copy
from collections import OrderedDict

import torch
from torch import nn

import configs
from models.predictor import Predictor
from models.utils import apply_mask
from models.vision_transformer import VisionTransformer


class JEPA(nn.Module):
  def __init__(self, config: configs.pretrain.Config, momentum_schedule, use_sdp_kernel=True):
    super().__init__()
    self.config = config
    self.momentum_schedule = momentum_schedule
    self.encoder = VisionTransformer(config, use_sdp_kernel=use_sdp_kernel)
    self.predictor = Predictor(config, use_sdp_kernel=use_sdp_kernel)
    self.target_encoder = copy.deepcopy(self.encoder)

    for param in self.target_encoder.parameters():
      param.requires_grad = False

  @torch.compiler.disable()
  def update_momentum_encoder(self):
    m = next(self.momentum_schedule)
    for param_z, param_h in zip(self.encoder.parameters(), self.target_encoder.parameters()):
      param_h.data = m * param_h.data + (1. - m) * param_z.data

  def forward(self, x, mask_encoder, mask_predictor):
    with torch.no_grad():
      self.update_momentum_encoder()
      # compute prediction targets
      h = self.target_encoder(x)
      h = apply_mask(h, mask_predictor)
    # encode unmasked patches
    z = self.encoder(x, mask_encoder)
    # predict masked patches
    z = self.predictor(z, mask_encoder, mask_predictor)
    loss = torch.mean(torch.abs(z - h))
    return loss

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
      eps=self.config.opt_eps,
      weight_decay=0.,
      fused=fused)

    return optimizer
