import argparse
import copy
import dataclasses
import logging.config
import pprint
from contextlib import nullcontext
from os import path, makedirs
from time import time

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch.nn import functional as F
from torch.utils.data import DataLoader

import configs
from data import transforms, utils as datautils
from data.datasets import PTB_XL
from data.utils import TensorDataset
from models import VisionTransformer, ViTClassifier
from utils.monitoring import AverageMeter, get_memory_usage, get_cpu_count
from utils.schedules import update_learning_rate_, cosine_schedule

TASKS = (
  'all',
  'diagnostic',
  'subdiagnostic',
  'superdiagnostic',
  'form',
  'rhythm',
  # custom tasks
  'ST-MEM',  # Na et al. (2024)
)
FOLDS = tuple(range(1, 11))

parser = argparse.ArgumentParser()
parser.add_argument('--data-dir', required=True, help='path to data directory')
parser.add_argument('--encoder', required=True, help='path to checkpoint or config file')
parser.add_argument('--out', default='eval', help='output directory')
parser.add_argument('--config', default='linear', help='path to config file or config name')
parser.add_argument('--dump', help='path to dump file (.npy) with raw ECG signals')
parser.add_argument('--amp', default='float32', choices=['bfloat16', 'float32'], help='automated mixed precision')
parser.add_argument('--task', choices=TASKS, default='all', help='task type')
parser.add_argument('--val-fold', choices=FOLDS, type=int, default=9, help='validation fold')
parser.add_argument('--test-fold', choices=FOLDS, type=int, default=10, help='test fold')
args = parser.parse_args()


def main():
  makedirs(args.out, exist_ok=True)
  logging.config.fileConfig('logging.ini')
  logger = logging.getLogger('app')

  dump_file = args.dump or f'{args.data_dir}.npy'
  if not path.isfile(dump_file):
    raise ValueError(f'Failed to find .npy data file. Attempted location: {dump_file}. '
                     f'Use `--dump` to specify location.')

  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  using_cuda = device.type == 'cuda'
  num_cpus = get_cpu_count()
  logger.debug(f'using {device} accelerator and {num_cpus} CPUs')

  if using_cuda:
    logger.debug('TF32 tensor cores are enabled')
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

  if args.amp == 'float32' or not using_cuda:  # don't use AMP on a CPU
    logger.debug('using float32 precision')
    auto_mixed_precision = nullcontext()
  elif args.amp == 'bfloat16':
    # bfloat16 preserves the range of float32, so it does not require scaling
    logger.debug('using bfloat16 with AMP')
    auto_mixed_precision = torch.cuda.amp.autocast(dtype=torch.bfloat16)
  else:
    raise ValueError('Failed to choose floating-point format.')

  if not path.isfile(args.config):
    # maybe config is the name of a default config file in configs/pretrain/
    config_file = path.join(path.dirname(configs.eval.__file__), f'{args.config}.yaml')
    if not path.isfile(config_file):
      raise ValueError(f'Failed to read configuration file {args.config}')
    args.config = config_file

  eval_config_dict = configs.load_config_file(args.config)
  logger.debug(f'loading configuration file from {args.config}\n'
               f'{pprint.pformat(eval_config_dict, compact=True, sort_dicts=False, width=120)}')

  # load checkpoint
  _, ext = path.splitext(args.encoder)
  if ext == '.yaml':
    logger.debug(f'loading encoder config from {args.encoder}')
    encoder_config_dict = configs.load_config_file(args.encoder)
    encoder_config = configs.pretrain.Config(**encoder_config_dict)
    model_state_dict = None
  else:
    logger.debug(f'loading encoder checkpoint from {args.encoder}')
    chkpt = torch.load(args.encoder, map_location='cpu')
    encoder_config_dict = chkpt['config']
    encoder_config = configs.pretrain.Config(**encoder_config_dict)
    if 'eval_config' in chkpt:  # continue fine-tuning the weights
      model_state_dict = chkpt['model']
    else:  # extract target encoder's weights from the checkpoint
      model_state_dict = {'encoder.' + k.removeprefix('target_encoder.'): v
                          for k, v in chkpt['model'].items()
                          if k.startswith('target_encoder.')}

  ptb_xl_task = args.task
  single_label = False
  if args.task == 'ST-MEM':
    ptb_xl_task = 'superdiagnostic'
    single_label = True

  # load labels
  logger.debug(f'setting up labels for task `{args.task}`')
  labels_df = PTB_XL.load_raw_labels(args.data_dir)
  labels_df = PTB_XL.compute_label_aggregations(labels_df, args.data_dir, ptb_xl_task)

  # load data
  logger.debug(f'loading data from {dump_file}')
  channel_size = PTB_XL.record_duration * encoder_config.sampling_frequency

  x = datautils.load_data_dump(
    dump_file=dump_file,
    transform=PreprocessECG(
      channel_size=channel_size,
      remove_baseline_wander=False),
    processes=num_cpus)

  x, labels_df, y, _ = PTB_XL.select_data(x, labels_df, ptb_xl_task, min_samples=0)
  if single_label:
    single_label_mask = y.sum(axis=1) == 1
    x, labels_df, y = x[single_label_mask], labels_df[single_label_mask], y[single_label_mask]
  y = torch.from_numpy(y).float()
  num_classes = y.shape[1]

  val_mask = (labels_df.strat_fold == args.val_fold).to_numpy()
  test_mask = (labels_df.strat_fold == args.test_fold).to_numpy()
  train_mask = ~(val_mask | test_mask)

  # normalize data
  mean = np.mean(x[train_mask], axis=(0, 1), keepdims=True, dtype=np.float32)
  std = np.std(x[train_mask], axis=(0, 1), keepdims=True, dtype=np.float32)
  transforms.normalize_(x, mean_std=(mean, std))
  x.clip(-5, 5, out=x)

  # ensure matching channels
  channel_order = datautils.get_channel_order(PTB_XL.channels, encoder_config.channels)
  x = x[:, :, channel_order]

  logger.debug(f'{get_memory_usage() / 1024 ** 3:,.2f}GB memory used after loading data')

  # initialize configs
  eval_config = configs.eval.Config(**eval_config_dict, num_classes=num_classes)
  if eval_config.use_register and encoder_config.num_registers == 0:
    logger.debug('adding a randomly initialized register to the encoder')
    encoder_config = dataclasses.replace(encoder_config, num_registers=1)

  if eval_config.dropout != encoder_config.dropout:
    logger.debug('overriding encoder dropout')
    encoder_config = dataclasses.replace(encoder_config, dropout=eval_config.dropout)

  if encoder_config.layer_scale_eps == 0 and eval_config.layer_scale_eps > 0:
    logger.debug('adding LayerScale to the encoder')
    encoder_config = dataclasses.replace(encoder_config, layer_scale_eps=eval_config.layer_scale_eps)

  if eval_config.crop_duration is not None:
    crop_size = int(eval_config.crop_duration * encoder_config.sampling_frequency)
    if eval_config.crop_stride is not None:
      crop_stride = int(eval_config.crop_stride * encoder_config.sampling_frequency)
    else:
      crop_stride = crop_size
  else:
    crop_size = None
    crop_stride = None

  train_loader = DataLoader(
    dataset=TensorDataset(
      data=x[train_mask],
      labels=y[train_mask],
      transform=TrainTransformECG(
        crop_size=crop_size)),
    batch_size=eval_config.batch_size,
    shuffle=True,
    drop_last=True,
    num_workers=2)

  def cycle(dataloader):
    while True:
      yield from dataloader

  train_iterator = cycle(train_loader)

  val_loader = DataLoader(
    dataset=TensorDataset(
      data=x[val_mask],
      labels=y[val_mask],
      transform=EvalTransformECG(
        crop_size=crop_size,
        crop_stride=crop_stride)),
    batch_size=eval_config.batch_size,
    num_workers=2)
  test_loader = DataLoader(
    dataset=TensorDataset(
      data=x[test_mask],
      labels=y[test_mask],
      transform=EvalTransformECG(
        crop_size=crop_size,
        crop_stride=crop_stride)),
    batch_size=eval_config.batch_size,
    num_workers=2)

  # setup hyperparameter schedules
  lr_schedule = cosine_schedule(
    total_steps=eval_config.steps,
    start_value=eval_config.learning_rate,
    final_value=eval_config.final_learning_rate,
    warmup_steps=eval_config.learning_rate_warmup_steps,
    warmup_start_value=1e-6)

  encoder = VisionTransformer(
    config=encoder_config,
    keep_registers=eval_config.use_register,
    use_sdp_kernel=using_cuda)
  model = ViTClassifier(encoder, eval_config, use_sdp_kernel=using_cuda).to(device)
  optimizer = model.get_optimizer(fused=using_cuda)

  if model_state_dict is not None:
    incompatible_keys = model.load_state_dict(model_state_dict, strict=False)
    for key in incompatible_keys.missing_keys:
      logger.debug(f'missing {key} in the encoder checkpoint')
    for key in incompatible_keys.unexpected_keys:
      logger.debug(f'unexpected {key} in the encoder checkpoint')

  step_time = AverageMeter()
  train_loss = AverageMeter()
  best_val_auc = float('-inf')
  best_val_predictions, val_targets = None, None
  best_step, best_chkpt = None, None

  for step in range(eval_config.steps):
    step_start = time()
    # update hyperparameters according to schedule
    update_learning_rate_(optimizer, next(lr_schedule))
    # forward pass
    x, y = (tensor.to(device) for tensor in next(train_iterator))
    with auto_mixed_precision:
      logits = model(x)
      if single_label:
        loss = F.cross_entropy(logits, y)
      else:
        loss = F.binary_cross_entropy_with_logits(logits, y)
    # backward pass
    loss.backward()
    if eval_config.gradient_clip > 0:
      torch.nn.utils.clip_grad_norm_(model.parameters(), eval_config.gradient_clip)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    # finalize train step
    step_end = time()
    step_time.update(step_end - step_start)
    train_loss.update(loss.item())
    # evaluation
    if (step + 1) % eval_config.checkpoint_interval == 0:
      val_logits, val_targets = [], []
      model.eval()
      with torch.inference_mode():
        for batch in val_loader:
          x, y = (tensor.to(device) for tensor in batch)
          if eval_config.crop_duration is not None:
            batch_size, num_crops, num_channels, channel_size = x.size()
            x = x.reshape(-1, num_channels, channel_size)
          logits = model(x)
          if eval_config.crop_duration is not None:
            logits = logits.reshape(batch_size, num_crops, eval_config.num_classes)
            logits = logits.mean(dim=1)  # aggregate crop predictions
          val_logits.append(logits.clone())
          val_targets.append(y.clone())
      model.train()
      if single_label:
        val_predictions = torch.cat(val_logits).softmax(dim=1).cpu().numpy()
      else:
        val_predictions = torch.cat(val_logits).sigmoid().cpu().numpy()
      val_targets = torch.cat(val_targets).cpu().numpy()
      val_auc = roc_auc_score(
        y_true=val_targets,
        y_score=val_predictions,
        average='macro')
      new_best_val_auc = val_auc > best_val_auc
      if new_best_val_auc:
        best_val_auc = val_auc
        best_val_predictions = val_predictions
        best_step = step
        best_chkpt = copy.deepcopy(model.state_dict())
      logger.info(f'[{step + 1:06d}] '
                  f'{"(*)" if new_best_val_auc else "   "} '
                  f'step_time {step_time.value:.4f} '
                  f'train_loss {train_loss.value:.4f} '
                  f'val_auc {val_auc:.4f}')
      step_time = AverageMeter()
      train_loss = AverageMeter()
      if step - best_step >= eval_config.early_stopping_patience:
        logging.info('stopping training early because validation AUC does not improve')
        break

  torch.save({
    'model': best_chkpt,
    'config': dataclasses.asdict(encoder_config),
    'eval_config': dataclasses.asdict(eval_config),
    'preprocess': {'mean': torch.from_numpy(mean.squeeze()),
                   'std': torch.from_numpy(std.squeeze())},
    'task': ptb_xl_task
  }, path.join(args.out, f'{args.task}_best_chkpt.pt'))

  # test model
  logger.info('loading best model checkpoint')
  model.load_state_dict(best_chkpt)

  test_logits, test_targets = [], []
  model.eval()
  with torch.inference_mode():
    for batch in test_loader:
      x, y = (tensor.to(device) for tensor in batch)
      if eval_config.crop_duration is not None:
        batch_size, num_crops, num_channels, channel_size = x.size()
        x = x.reshape(-1, num_channels, channel_size)
      logits = model(x)
      if eval_config.crop_duration is not None:
        logits = logits.reshape(batch_size, num_crops, eval_config.num_classes)
        logits = logits.mean(dim=1)  # aggregate crop predictions
      test_logits.append(logits.clone())
      test_targets.append(y.clone())
  if single_label:
    test_predictions = torch.cat(test_logits).softmax(dim=1).cpu().numpy()
  else:
    test_predictions = torch.cat(test_logits).sigmoid().cpu().numpy()
  test_targets = torch.cat(test_targets).cpu().numpy()
  test_auc = roc_auc_score(
      y_true=test_targets,
      y_score=test_predictions,
      average='macro')
  logger.info(f'test_auc {test_auc:.4f}')
  np.savez(path.join(args.out, f'{args.task}_predictions.npz'),
           val_targets=val_targets, val_predictions=best_val_predictions,
           test_targets=test_targets, test_predictions=test_predictions)


class PreprocessECG:
  def __init__(self, channel_size=None, remove_baseline_wander=False):
    self.channel_size = channel_size
    self.remove_baseline_wander = remove_baseline_wander

  def __call__(self, x):
    channel_size, num_channels = x.shape
    if self.remove_baseline_wander:
      x = transforms.highpass_filter(x, fs=PTB_XL.sampling_frequency)
    if self.channel_size is not None and self.channel_size != channel_size:
      x = transforms.resample(x, self.channel_size)
    return x


class TrainTransformECG:   # called whenever dataloader accesses the data
  def __init__(self, crop_size=None):
    self.crop_size = crop_size

  def __call__(self, x):
    if self.crop_size is not None:
      x = transforms.random_crop(x, self.crop_size)
    x = x.transpose()  # channels first
    x = torch.from_numpy(x).float()
    return x


class EvalTransformECG:  # called whenever dataloader accesses the data
  def __init__(self, crop_size=None, crop_stride=None):
    self.crop_size = crop_size
    self.crop_stride = crop_stride or crop_size

  def __call__(self, x):
    if self.crop_size is not None:
      x = strided_crops(x, self.crop_size, self.crop_stride)
      x = np.swapaxes(x, 1, 2)  # channels first
    else:
      x = x.transpose()  # channels first
    x = torch.from_numpy(x).float()
    return x


def strided_crops(x, size, stride):  # x: (channel_size, num_channels)
  channel_size, num_channels = x.shape
  crop_starts = range(0, channel_size - size + 1, stride)
  num_crops = len(crop_starts)
  x_ = np.empty((num_crops, size, num_channels), dtype=x.dtype)
  for i, start in enumerate(crop_starts):
    x_[i] = x[start:start + size]
  return x_


if __name__ == '__main__':
  main()
