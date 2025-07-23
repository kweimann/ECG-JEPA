from data.datasets.challenge_2021 import (
  ChapmanShaoxing,
  CPSC2018,
  CPSC2018Extra,
  Georgia,
  Ningbo,
  PTB,
  StPetersburg
)
from data.datasets.code_15 import CODE15
from data.datasets.mimic_iv_ecg import MIMIC_IV_ECG
from data.datasets.ptb_xl import PTB_XL

DATASETS = {
  'chapman-shaoxing': ChapmanShaoxing,
  'cpsc': CPSC2018,
  'cpsc-extra': CPSC2018,
  'georgia': Georgia,
  'ningbo': Ningbo,
  'ptb': PTB,
  'st-petersburg': StPetersburg,
  'code-15': CODE15,
  'mimic-iv-ecg': MIMIC_IV_ECG,
  'ptb-xl': PTB_XL
}
