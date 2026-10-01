"""# Imports"""

import argparse
import gc
import sys

import torch

from hyperparameters.hyperparameters import Hyperparameters
from ExperimentBuilder import ExperimentBuilder

parser = argparse.ArgumentParser(description="Training per Fold")
parser.add_argument('--net', type=str, required=True, help='net')
parser.add_argument('--dataset', type=str, required=True, help='dataset')
parser.add_argument('--label', type=str, required=True, help='label')
parser.add_argument('--fold', type=int, required=True, help='fold')
parser.add_argument('--local_time', type=str, required=True, help="local_time")
args = parser.parse_args()

torch.multiprocessing.set_sharing_strategy('file_system')
torch.set_float32_matmul_precision("high")

"""# Checking device"""

print('CUDA available?', torch.cuda.is_available())
print('Current device:', torch.cuda.current_device())
print('Device name:', torch.cuda.get_device_name(0))

"""# Fit"""

approach = 'resized'
_datamodule = 'Generic3dDataModule'
transforms = 'Generic3dTransforms'
_task = 'Generic3dSegmentationTask'

net = args.net
dataset = args.dataset
label = args.label
k = args.fold
local_time = args.local_time

# Skip
# if (label != 'spinal_cord' and label != 'trachea') and net == 'VNet':
#     print(f"Skip {net} - label {label}")
#     sys.exit(0)

hu_range = {
    'structseg_thorax': [-310, 400],
    'multiorgan_ct_btcv': [-150, 250],
    'structseg_head': [-200, 700],
}

if '_' in approach:
    approach_tag = approach.partition('_')[0]
else:
    approach_tag = approach

dataset_path = f'../data/datasets/{dataset}_{approach_tag}'
labels_list = ExperimentBuilder._get_dataset_labels(dataset_path)

if label not in labels_list:
    print(f"Unknown label: '{label}'")
    sys.exit(1)

print(f"\nFit: {net} - {dataset}: {label}\n")

if net == 'UNETR':
    net_flag = 'att'
else:
    net_flag = 'cnn'

hyperparameters = Hyperparameters()
hps = hyperparameters.get_hyperparameters(f'{net_flag}_{dataset}')[label]

print('Hyperparameters:')
for key, v in hps.items():
    print(key, v)

"""Experiment setup"""

# Generic3dSegmentationTask
# batch_size = 4
# accumulate_grad_batches = 1
# num_workers = 8
# log_every_n_steps = 10
# check_val_every_n_epoch = 1

# PatchCropSegmentationTask
# num_samples = 2
# sw_batch_size = 8
# batch_size = 2
# accumulate_grad_batches = 1
# num_workers = 4
# log_every_n_steps = 20
# check_val_every_n_epoch = 10

# PatchSubsamplingSegmentationTask
# batch_size = 4 ou 8
# accumulate_grad_batches = 1
# num_workers = 8
# log_every_n_steps = 10 ou 5
# check_val_every_n_epoch = 10

setup = {
    'experiment_id': f'official_sibgrapi_{approach}_{net}_{dataset}_{label}_{local_time}', 
    'accelerator': 'gpu' if torch.cuda.is_available() else 'cpu',
    'devices': 1 if torch.cuda.is_available() else 'auto',
    'dataset_name': dataset,
    'dataset_path': dataset_path,
    'target_label': label,
    'datamodule': _datamodule,
    'transforms_set': transforms,
    'hu_range': hu_range[dataset],
    'task': _task,
    'architecture': net,
    'in_channels': 1,
    'num_classes': 2,
    'img_size': (128, 128, 128),
    'loss': 'DiceCELoss',
    'seed': 42,
    'batch_size': 4, 
    'num_workers': 8,
    'one_cycle_lr': True,
    'base_lr': hps['base_lr'],
    'weight_decay': hps['weight_decay'],
    'max_lr': hps['max_lr'],
    'pct_start': hps['pct_start'],
    'div_factor': hps['div_factor'],
    'final_div_factor': hps['final_div_factor'],
    'max_epochs': 200,
    'k-fold': 5, 
    'save_top_k': 1,
    'log_every_n_steps': 10, 
    'check_val_every_n_epoch': 10, 
    'monitor_metric': 'IoU/val',
    'monitor_mode': 'max',
    'num_imgs_to_plot': 1
}

"""# Experiment builder"""

experiment = ExperimentBuilder(setup, base_outdir_path=f"logs/{setup['dataset_name']}/{setup['target_label']}")

"""# Main"""

print(f'\nTraining started - Fold {k}')

gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

datamodule = experiment.get_datamodule(k)
model_architecture = experiment.get_model_architecture()
loss_function = experiment.get_loss_function()
task = experiment.get_task(model_architecture, loss_function)
trainer = experiment.get_trainer(k)

trainer.fit(task, datamodule=datamodule)

del datamodule
del model_architecture
del loss_function
del task
del trainer

gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

print(f'\nTraining completed - Fold {k}\n')
