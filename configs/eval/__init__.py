from dataclasses import dataclass
from typing import Optional


@dataclass
class Config:
  # data
  crop_duration: Optional[float]  # seconds
  crop_stride: Optional[float]  # seconds
  # model architecture
  num_classes: int
  use_register: bool
  attn_pooling: bool
  layer_scale_eps: float
  bias: bool
  dropout: float
  frozen: bool
  # training
  steps: int
  batch_size: int
  learning_rate: float
  final_learning_rate: float
  learning_rate_warmup_steps: int
  weight_decay: float
  opt_betas: tuple[float, float]
  gradient_clip: float
  checkpoint_interval: int
  early_stopping_patience: int
