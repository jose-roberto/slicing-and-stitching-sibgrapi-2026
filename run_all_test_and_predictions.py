"""# Imports"""

import argparse
import gc
import glob
import numpy as np
import os
import pandas as pd
from scipy import stats
import sys

import torch

import nibabel as nib

from ExperimentBuilder import ExperimentBuilder

parser = argparse.ArgumentParser(description="Prediction per Fold")
parser.add_argument('--ckpt_path', type=str, required=True, help='ckpt_path')
parser.add_argument('--nets', nargs='+', required=True, help='nets')
parser.add_argument('--dataset', type=str, required=True, help='dataset')
parser.add_argument('--labels', nargs='+', required=True, help='labels')
args = parser.parse_args()

torch.multiprocessing.set_sharing_strategy('file_system')
torch.set_float32_matmul_precision("high")

"""# Checking device"""

print('CUDA available?', torch.cuda.is_available())
print('Current device:', torch.cuda.current_device())
print('Device name:', torch.cuda.get_device_name(0))

"""# Experiment setup"""

approach = 'patch_subsampling'
_datamodule = 'PatchSubsamplingDataModule'
transforms = 'PatchSubsamplingTransforms'
_task = 'PatchSubsamplingSegmentationTask'

ckpt_path = args.ckpt_path
print(ckpt_path)
ckpt_id = os.path.basename(ckpt_path)
print(ckpt_id)

net = next((n for n in args.nets if n in ckpt_id))
dataset_name = args.dataset
label = next((l for l in args.labels if l in ckpt_id))

hu_range = {
    'structseg_thorax': [-310, 400],
    'multiorgan_ct_btcv': [-150, 250],
    'structseg_head': [-200, 700],
}

if '_' in approach:
    approach_tag = approach.partition('_')[0]
else:
    approach_tag = approach

# Skip
# if label != 'both_lungs':
#     print(f"Skip {label}")
#     sys.exit(0)

setup = {
    'experiment_id': ckpt_id,
    'accelerator': 'gpu' if torch.cuda.is_available() else 'cpu',
    'devices': 1 if torch.cuda.is_available() else 'auto',
    'dataset_name': dataset_name,
    'dataset_path': f'../data/datasets/{dataset_name}_{approach_tag}',
    'target_label': label,
    'datamodule': _datamodule,
    'transforms_set': transforms,
    'hu_range': hu_range[dataset_name],
    'task': _task,
    'architecture': net,
    'in_channels': 1,
    'num_classes': 2,
    'img_size': (128, 128, 128),
    'loss': 'DiceCELoss',
    'seed': 42,
    'batch_size': 4,
    'num_workers': 8,
    'one_cycle_lr': None,
    'base_lr': None,
    'weight_decay': None,
    'max_lr': None,
    'pct_start': None,
    'div_factor': None,
    'final_div_factor': None,
    'max_epochs': None,
    'k-fold': 5,
    'save_top_k': None,
    'monitor_metric': None,
    'monitor_mode': None,
    'log_every_n_steps': None,
    'check_val_every_n_epoch': None,
    'num_imgs_to_plot': None
}

prediction_path = f'predictions/{setup["dataset_name"]}/{setup["target_label"]}/{setup["experiment_id"].replace("ckpt", "pred")}'
os.makedirs(os.path.join(prediction_path), exist_ok=True)

"""# Experiment builder"""

experiment = ExperimentBuilder(setup, prediction_path, save_setup=False)

"""# Predict"""

all_folds_results = []

for k in range(setup['k-fold']):

    print(f' Fold {k}')

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    prediction_output_path = f'{prediction_path}/fold_{k}'
    os.makedirs(prediction_output_path, exist_ok=True)

    datamodule = experiment.get_datamodule(k)

    datamodule.setup('test')

    test_loader = datamodule.test_dataloader()

    checkpoint_path = glob.glob(f'{ckpt_path}/fold_{k}/*.ckpt')[0]

    model_architecture = experiment.get_model_architecture()

    loss_function = experiment.get_loss_function()

    task = experiment.load_from_checkpoint(checkpoint_path, setup, model_architecture, loss_function)

    task.prediction_output_path = prediction_output_path

    trainer = experiment.get_test_trainer()

    fold_metrics = trainer.test(task, dataloaders=test_loader)[0]

    fold_metrics['Fold'] = k
    fold_metrics['Experiment_ID'] = setup["experiment_id"]

    all_folds_results.append(fold_metrics)

    _ = trainer.predict(task, dataloaders=test_loader)

    del datamodule
    del model_architecture
    del loss_function
    del task
    del trainer
    
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

df_results = pd.DataFrame(all_folds_results)

df_results.replace([np.inf, -np.inf], np.nan, inplace=True)

"""# Stats"""

if 'HausdorffDistance95/test' in df_results.columns and 'HausdorffDistance/test' in df_results.columns:
    df_results['HD95_HD_Ratio/test'] = df_results['HausdorffDistance95/test'] / df_results['HausdorffDistance/test']

cols = ['Fold', 'Experiment_ID'] + [c for c in df_results.columns if c not in ['Fold', 'Experiment_ID', 'HD95_HD_Ratio/test', 'SurfaceDistance/test']]

if 'HD95_HD_Ratio/test' in df_results.columns:
    cols.append('HD95_HD_Ratio/test')
if 'SurfaceDistance/test' in df_results.columns:
    cols.append('SurfaceDistance/test')

df_results = df_results[cols]

percentage_metrics = ['IoU/test', 'Dice/test', 'Precision/test', 'Recall/test']
distance_metrics = ['HausdorffDistance/test', 'HausdorffDistance95/test', 'HD95_HD_Ratio/test', 'SurfaceDistance/test']

print("\nT-Student - Confidence Intervals")

metric_cols = [c for c in df_results.columns if c not in ['Fold', 'Experiment_ID']]

n_folds = setup['k-fold']
degrees_of_freedom = n_folds - 1
confidence_level = 0.95

t_critical = stats.t.ppf((1 + confidence_level) / 2, degrees_of_freedom)

summary_stats = []

for col in metric_cols:

    values = pd.to_numeric(df_results[col]) if df_results[col].dtype == object else df_results[col]
    
    mean = values.mean()
    std_dev = values.std(ddof=1) 
    std_error = std_dev / np.sqrt(n_folds)
    
    margin_of_error = t_critical * std_error

    lower_bound = mean - margin_of_error
    upper_bound = mean + margin_of_error

    summary_stats.append({
        'Metric': col,
        'Mean': mean,
        'Std': std_dev,
        'Error': std_error,
        'CI_95%_Lower': lower_bound,
        'CI_95%_Upper': upper_bound,
        'Margin_Error': margin_of_error 
    })

df_summary = pd.DataFrame(summary_stats)

"""# Format and save"""

df_results_fmt = df_results.copy()

for col in percentage_metrics:
    if col in df_results_fmt.columns:
        df_results_fmt[col] = df_results_fmt[col].map(lambda x: f"{x:.4f}".replace('.', ','))

for col in distance_metrics:
    if col in df_results_fmt.columns:
        df_results_fmt[col] = df_results_fmt[col].map(lambda x: f"{x:.2f}".replace('.', ','))

csv_filename = os.path.join(prediction_path, 'metrics_summary.csv')
df_results_fmt.to_csv(csv_filename, index=False)

print(f"\nMetrics saved in: {csv_filename}")

def format_final_stats(row):
    metric = row['Metric']

    fmt_report = lambda x: f"{x:.4f}".replace('.', ',')
    
    if metric in percentage_metrics:
        fmt_main = lambda x: f"{x:.4f}".replace('.', ',')
    elif metric in distance_metrics:
        fmt_main = lambda x: f"{x:.2f}".replace('.', ',')
        fmt_report = lambda x: f"{x:.2f}".replace('.', ',')
    else:
        fmt_main = lambda x: str(x)

    return pd.Series({
        'Metric': metric,
        'Mean': fmt_main(row['Mean']),
        'Std': fmt_main(row['Std']),
        'Error': fmt_main(row['Error']),
        'CI_95%_Lower': fmt_main(row['CI_95%_Lower']),
        'CI_95%_Upper': fmt_main(row['CI_95%_Upper']),
        'Report': f"{fmt_report(row['Mean'])} ± {fmt_report(row['Margin_Error'])}"
    })

df_summary_final = df_summary.apply(format_final_stats, axis=1)

summary_filename = os.path.join(prediction_path, 'metrics_confidence_intervals_summary.csv')
df_summary_final.to_csv(summary_filename, index=False)

print(f"Confidence intervals saved in: {summary_filename}")