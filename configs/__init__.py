import yaml

from configs import (
  pretrain,
  eval
)


def load_config_file(config_file, **kwargs):
  with open(config_file) as fh:
    config_dict = yaml.safe_load(fh)
  config_dict = {**config_dict, **kwargs}  # override file config with kwargs
  return config_dict
