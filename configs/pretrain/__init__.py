from dataclasses import dataclass


@dataclass
class Config:
  # data
  sampling_frequency: int
  channels: tuple[str, ...]
  channel_size: int
  patch_size: int
  min_block_size: int
  min_keep_ratio: float
  max_keep_ratio: float
  datasets: dict[str, float]
  # model architecture
  dim: int
  depth: int
  num_heads: int
  pred_dim: int
  pred_depth: int
  pred_num_heads: int
  mlp_ratio: float
  qkv_bias: bool
  dropout: float
  attn_dropout: float
  num_registers: int
  bias: bool
  norm_eps: float
  layer_scale_eps: float
  # training
  steps: int
  batch_size: int
  encoder_momentum: float
  final_encoder_momentum: float
  learning_rate: float
  final_learning_rate: float
  learning_rate_warmup_steps: int
  weight_decay: float
  final_weight_decay: float
  opt_betas: tuple[float, float]
  opt_eps: float
  gradient_clip: float
  gradient_accumulation_steps: int
  checkpoint_interval: int

  @property
  def num_channels(self):
    return len(self.channels)
