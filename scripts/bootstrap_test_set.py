import argparse
import os
from os import path

import numpy as np
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--predictions', required=True, help='Path to the predictions file or directory')
parser.add_argument('--seed', type=int, help='random state')
args = parser.parse_args()

files = []

if path.isfile(args.predictions):
  files.append(args.predictions)
elif path.isdir(args.predictions):
  for root, _, filenames in os.walk(args.predictions):
    for file in filenames:
      if file.endswith('.npz'):
        files.append(path.join(root, file))

y_true = None
y_preds = []

for file in files:
  data = np.load(file)
  if y_true is None:
    y_true = data['test_targets']
  else:
    assert np.array_equal(y_true, data['test_targets'])
  y_preds.append(data['test_predictions'])

n_bootstraps = 100
rng = np.random.RandomState(args.seed)

all_bootstrapped_scores = []
per_model_results = []

for file, y_pred in zip(files, y_preds):
  bootstrapped_scores = []
  with tqdm(total=n_bootstraps, leave=False, desc='Bootstrapping') as pbar:
    while len(bootstrapped_scores) < n_bootstraps:
      indices = rng.choice(len(y_true), size=len(y_true), replace=True)

      y_true_sample = y_true[indices]
      y_pred_sample = y_pred[indices]

      try:
          score = roc_auc_score(y_true_sample, y_pred_sample, average="macro")
          bootstrapped_scores.append(score)
      except ValueError:
          continue

      pbar.update()

  lower, upper = np.percentile(bootstrapped_scores, [2.5, 97.5])
  mean_score = np.mean(bootstrapped_scores)

  print(f"{file} AUC: {mean_score:.3f} 95% CI: ({lower:.3f}, {upper:.3f})")

  per_model_results.append((mean_score, lower, upper))
  all_bootstrapped_scores.extend(bootstrapped_scores)

per_model_means, all_ci_lowers , all_ci_uppers  = zip(*per_model_results)
overall_mean = np.mean(per_model_means)
overall_std = np.std(per_model_means)
ci_range_lower = min(all_ci_lowers)
ci_range_upper = max(all_ci_uppers)

print("\n=== Summary ===")
print(f"Mean AUC across models: {overall_mean:.3f} ± {overall_std:.3f}")
print(f"Range of 95% CIs across models: ({ci_range_lower:.3f}, {ci_range_upper:.3f})")

global_mean, global_std = np.mean(all_bootstrapped_scores), np.std(all_bootstrapped_scores)
global_ci_lower, global_ci_upper = np.percentile(all_bootstrapped_scores, [2.5, 97.5])

print("\n=== Pooled Bootstrap Summary ===")
print(f"Pooled AUC (macro): {global_mean:.3f} ± {global_std:.3f}")
print(f"95% CI (pooled across models): ({global_ci_lower:.3f}, {global_ci_upper:.3f})")
