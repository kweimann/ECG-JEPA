import os

import psutil


class AverageMeter:
  def __init__(self):
    self.value = None
    self._sum = 0
    self._count = 0

  def update(self, value, count=1):
    self._sum += value
    self._count += count
    self.value = self._sum / self._count
    return self.value


def get_cpu_count():
  cpus_per_task = os.getenv('SLURM_CPUS_PER_TASK')
  if cpus_per_task:
    return int(cpus_per_task)
  else:
    return os.cpu_count()


def get_memory_usage():
  process = psutil.Process()
  memory_usage = process.memory_info().rss
  for child_process in process.children(recursive=True):
    memory_usage += child_process.memory_info().rss
  return memory_usage
