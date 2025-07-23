# ECG-JEPA

---

This is official implementation of the paper: "Self-Supervised Pre-Training with Joint-Embedding Predictive Architecture Boosts ECG Classification Performance" (https://arxiv.org/abs/2410.13867).

### Results

We pre-train Vision Transformers on various ECG datasets using the [JEPA](https://arxiv.org/abs/2301.08243) framework, then we fine-tune the
pre-trained models on the PTB-XL database. The table below compares test performances on the all statements task. 
The scores are average test AUCs computed over 10 runs, with standard deviation in brackets (0.0xx).

| Model         | Method                    | All \[Fine-tune\] | All \[Linear\] | Source                                                                          |
|---------------|---------------------------|-------------------|----------------|---------------------------------------------------------------------------------|
| inception1d   | Random Init               | 0.925(08)         | —              | [helme/ecg_ptbxl_benchmarking](https://github.com/helme/ecg_ptbxl_benchmarking) |
| xresnet1d50   | Random Init               | 0.924(05)         | 0.721(16)      | [hhi-aml/ecg-selfsupervised](https://github.com/hhi-aml/ecg-selfsupervised)     |
| 4FC+2LSTM+2FC | Random Init               | 0.932(03)         | 0.711(07)      | [hhi-aml/ecg-selfsupervised](https://github.com/hhi-aml/ecg-selfsupervised)     |
| ViT-B         | Random Init               | 0.837(17)         | 0.867(05)      | This repository                                                                 |
| ViT-S         | Random Init               | 0.883(04)         | 0.833(06)      | This repository                                                                 |
| ViT-XS        | Random Init               | 0.911(04)         | 0.815(10)      | This repository                                                                 |
| xresnet1d50   | SimCLR                    | 0.927(03)         | 0.883(03)      | [hhi-aml/ecg-selfsupervised](https://github.com/hhi-aml/ecg-selfsupervised)     |
| xresnet1d50   | BYOL                      | 0.929(02)         | 0.878(02)      | [hhi-aml/ecg-selfsupervised](https://github.com/hhi-aml/ecg-selfsupervised)     |
| 4FC+2LSTM+2FC | CPC (CinC2020)            | 0.942(01)         | 0.927(01)      | [hhi-aml/ecg-selfsupervised](https://github.com/hhi-aml/ecg-selfsupervised)     |
| 4FC+2LSTM+2FC | CPC (CinC2020 w/o PTB-XL) | 0.940(02)         | 0.919(01)      | [hhi-aml/ecg-selfsupervised](https://github.com/hhi-aml/ecg-selfsupervised)     |
| S4            | CPC (CinC2021)            | **0.945(02)**     | -              | [tmehari/ssm_ecg](https://github.com/tmehari/ssm_ecg)                           |
| ViT-B         | JEPA (All)                | 0.940(01)         | 0.935(01)      | This repository                                                                 |
| ViT-S         | JEPA (All)                | **0.945(01)**     | 0.938(02)      | This repository                                                                 |
| ViT-S         | JEPA (MIMIC-IV-ECG)       | 0.944(01)         | **0.940(02)**  | This repository                                                                 |
| ViT-S         | JEPA (PTB-XL)             | 0.930(01)         | 0.926(02)      | This repository                                                                 |
| ViT-XS        | JEPA (All)                | 0.939(00)         | 0.933(02)      | This repository                                                                 |
| ViT-XS        | JEPA (MIMIC-IV-ECG)       | 0.943(01)         | 0.933(03)      | This repository                                                                 |
| ViT-XS        | JEPA (PTB-XL)             | 0.940(01)         | 0.931(02)      | This repository                                                                 |

---

The second table shows additional comparisons with the [ST-MEM](https://arxiv.org/abs/2402.09450) method that is based on the [Masked Autoencoders](https://arxiv.org/abs/2111.06377) (MAE) with ViT-B as the backbone.
The authors of ST-MEM evaluate their model and other pre-training techniques on the diagnostic superclasses task with only those records that have a single label. 
Furthermore, they use their own train-val-test split. We report our results on the same task, but using the recommended train-val-test split.

| Model         | Method                 | Superdiagnostic<br>(Single Label) \[Fine-tune\] | Superdiagnostic<br>(Single Label) \[Linear\] | Source                           |
|---------------|------------------------|-------------------------------------------------|----------------------------------------------|----------------------------------|
| ViT-B         | MoCo v3                | 0.913(02)                                       | 0.739(06)                                    | https://arxiv.org/abs/2402.09450 |
| ViT-B         | CMSC                   | 0.877(03)                                       | 0.797(38)                                    | https://arxiv.org/abs/2402.09450 |
| ViT-B         | MTAE                   | 0.910(01)                                       | 0.807(06)                                    | https://arxiv.org/abs/2402.09450 |
| ViT-B         | MTAE+RLM               | 0.911(04)                                       | 0.806(05)                                    | https://arxiv.org/abs/2402.09450 |
| ViT-B         | MLAE                   | 0.915(01)                                       | 0.779(08)                                    | https://arxiv.org/abs/2402.09450 |
| ViT-B         | ST-MEM                 | 0.933(03)                                       | 0.838(11)                                    | https://arxiv.org/abs/2402.09450 |
| 4FC+2LSTM+2FC | CPC (w/ entire PTB-XL) | 0.934(02)                                       | —                                            | https://arxiv.org/abs/2402.09450 |
| ViT-B         | JEPA (All)             | 0.928(03)                                       | 0.920(02)                                    | This repository                  |
| ViT-S         | JEPA (All)             | **0.935(02)**                                   | **0.928(03)**                                | This repository                  |
| ViT-S         | JEPA (MIMIC-IV-ECG)    | 0.932(02)                                       | 0.921(03)                                    | This repository                  |
| ViT-S         | JEPA (PTB-XL)          | 0.929(02)                                       | 0.917(01)                                    | This repository                  |
| ViT-XS        | JEPA (All)             | 0.930(01)                                       | 0.924(02)                                    | This repository                  |
| ViT-XS        | JEPA (MIMIC-IV-ECG)    | 0.928(02)                                       | 0.920(02)                                    | This repository                  |
| ViT-XS        | JEPA (PTB-XL)          | 0.928(02)                                       | 0.919(02)                                    | This repository                  |

### Reproduction: Pre-training

---

Here, we pre-train ViTS on just the MIMIC-IV-ECG dataset. First, we need to dump all records into a NumPy array for 
easy access. We detail how other ECG datasets are prepared in the further section. Then, we run `pretrain.py` to start the pre-training. 
The `--data` argument accepts a list of datasets and their locations in the following format: `dataset_name=path`.
The weighting of the datasets that is used to sample records must be provided in the configuration file.
All pre-training configurations are available in [configs/pretrain](configs/pretrain).

```shell
# create a .npy dataset (only once)
python -m scripts.dump_data --data-dir "/path/to/mimic-iv-ecg" --verbose
# pre-train ViT encoder
python -m pretrain \
  --data "mimic-iv-ecg=/path/to/mimic-iv-ecg.npy" \
  --out "pretrain-output-dir" \
  --config "ViTS_mimic" \
  --amp "bfloat16"
```

### Reproduction: Evaluation through fine-tuning

---

In this example, we first prepare the PTB-XL dataset for easy access by dumping all records into a NumPy array. 
Then, we run the `finetune.py` script to fine-tune a pre-trained Vision Transformer on the all statements task
from the PTB-XL database. The `--encoder` argument also accepts a path to an encoder config file (.yaml). 
In that case, the Vision Transformer will be trained from scratch. All evaluation configs are available in 
[configs/eval](configs/eval).

```shell
# create a .npy dataset (only once)
python -m scripts.dump_data --data-dir "/path/to/ptb-xl" --verbose
# linear evaluation protocol on all statements
python -m finetune \
  --data-dir "/path/to/ptb-xl" \
  --encoder "/path/to/checkpoint.pt" \
  --out "linear-output-dir" \
  --config "linear" \
  --task "all"
# end-to-end fine-tuning from linear checkpoint on all statements
python -m finetune \
  --data-dir "/path/to/ptb-xl" \
  --encoder "linear-output-dir/all_best_chkpt.pt" \
  --out "finetune-output-dir" \
  --config "finetune_after_linear" \
  --task "all"
```

Below, we directly fine-tune a pre-trained model on the ST-MEM task.

```shell
python -m finetune \
  --data-dir "/path/to/ptb-xl" \
  --encoder "/path/to/checkpoint.pt" \
  --out "finetune-output-dir" \
  --config "finetune" \
  --task "ST-MEM"
```

### ECG Datasets

---

Here is the list of all ECG datasets that we use for pre-training.

| Dataset                     | Records       | Seconds        | Source                                 |
|-----------------------------|---------------|----------------|----------------------------------------|
| MIMIC-IV-ECG                | 800,035       | 8,000,350      | https://doi.org/10.13026/4nqg-sb35     |
| CODE-15                     | 128,033       | 1,311,060      | https://doi.org/10.5281/zenodo.4916206 |
| PTB-XL (training partition) | 17,439        | 174,390        | https://doi.org/10.13026/kfzx-aw45     |
| Chapman-Shaoxing            | 10,247        | 102,470        | https://doi.org/10.13026/34va-7q14     |
| CPSC                        | 6,867         | 109,585        | https://doi.org/10.13026/34va-7q14     |
| CPSC-Extra                  | 3,441         | 54,819         | https://doi.org/10.13026/34va-7q14     |
| Georgia                     | 10,292        | 102,920        | https://doi.org/10.13026/34va-7q14     |
| Ningbo                      | 34,905        | 349,050        | https://doi.org/10.13026/34va-7q14     |
| PTB                         | 516           | 57,150         | https://doi.org/10.13026/34va-7q14     |
| St-Petersburg               | 74            | 133,200        | https://doi.org/10.13026/34va-7q14     |
| **Total**                   | **1,011,849** | **10,394,994** |

Below we outline how to preprocess and dump every dataset. 
Generally, we dump the datasets in their original form.
However, we removed the baseline wander from CODE-15 and St-Petersburg to facilitate stable pre-training.

```shell
# datasets: MIMIC-IV-ECG, Chapman-Shaoxing, CPSC, CPSC-Extra, Georgia, Ningbo, PTB
python -m scripts.dump_data --data-dir "/path/to/data" --verbose
```
```shell
# datasets: CODE-15, St-Petersburg
python -m scripts.dump_data --data-dir "/path/to/data" --verbose
python -m scripts.preprocess_data_dump --data "/path/to/data.npy" --interpolate-nans --remove-baseline-wander
```
```shell
# PTB-XL (full dataset for fine-tuning)
python -m scripts.dump_data --data-dir "/path/to/data" --verbose
# PTB-XL (training partition for pre-training; after data was dumped)
python -m scripts.split_ptb_xl --data-dir "/path/to/data" --folds 1 2 3 4 5 6 7 8
```

### Installation

1. Install `Python 3.11`
2. Create virtual environment: 

   `$ python -m venv <venv>`

3. Install packages:
    
    `$ source <venv>/bin/activate`

    `$ pip install -r requirements.txt`

### Archived Repository

@ZIB: https://git.zib.de/bzfweima/ecg-jepa