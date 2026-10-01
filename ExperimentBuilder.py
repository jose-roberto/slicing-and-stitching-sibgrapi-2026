import json
import os

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, RichProgressBar
from pytorch_lightning.loggers import TensorBoardLogger

from architectures.vnet import VNet
from architectures.resunet50 import ResUNet50
from architectures.highresnet import HighRes3DNet
from architectures.resnetmed import ResNetMed3D
from architectures.skipdensenet import SkipDenseNet3D
from architectures.unetr import UNETR
from transforms.generic_3d_transforms import Generic3dTransforms
from transforms.patch_crop_transforms import PatchCropTransforms
from transforms.patch_subsampling_transforms import PatchSubsamplingTransforms
from datamodules.generic_3d_datamodule import Generic3dDataModule
from datamodules.patch_crop_datamodule import PatchCropDataModule
from datamodules.patch_subsampling_datamodule import PatchSubsamplingDataModule
from loss_functions.wce import WCELoss
from tasks.generic3d_segmentation_task import Generic3dSegmentationTask
from tasks.patch_crop_segmentation_task import PatchCropSegmentationTask
from tasks.patch_subsampling_segmentation_task import PatchSubsamplingSegmentationTask
from callbacks.pruner import PruningCallback

from monai.networks.nets import VNet as MonaiVNet
from monai.networks.nets import UNETR as MonaiUNETR
from monai.losses import DiceLoss, DiceCELoss

class ExperimentBuilder():
    
    def __init__(self, setup, base_outdir_path, save_setup=True):
        
        super().__init__()

        self.setup = setup

        self.seed = setup['seed']
        pl.seed_everything(setup['seed'], workers=True)

        self.experiment_id = setup['experiment_id']
        self.outdir_path = os.path.join(base_outdir_path, self.experiment_id)

        if save_setup:
            os.makedirs(self.outdir_path, exist_ok=True)
            self._save_setup_json()
        
        self.accelerator = setup['accelerator']
        self.devices = setup['devices']

        self.dataset_path = setup['dataset_path']
        self.dataset_name = setup['dataset_name']
        self.target_label = setup['target_label']
        self.datamodule = setup['datamodule']

        self.task = setup['task']
        
        self.transforms_set = setup['transforms_set']
        self.hu_range = setup['hu_range']
        self.train_transforms = None
        self.val_transforms = None
        
        self.architecture = setup['architecture']
        self.in_channels = setup['in_channels']
        self.num_classes = setup['num_classes']
        self.img_size = setup['img_size']

        self.loss = setup['loss']

        self.batch_size = setup['batch_size']
        self.num_workers = setup['num_workers']
        self.max_epochs = setup['max_epochs']
        self.save_top_k = setup['save_top_k']
        self.log_every_n_steps = setup['log_every_n_steps']
        self.check_val_every_n_epoch = setup['check_val_every_n_epoch']
        self.monitor_metric = setup['monitor_metric']
        self.monitor_mode = setup['monitor_mode']

        self.labels_list = self._get_dataset_labels(self.dataset_path)
        self.n_labels = self._unique_elements(self.labels_list)

    def _save_setup_json(self):
        
        setup_path = os.path.join(self.outdir_path, 'setup.json')
        
        with open(setup_path, 'w', encoding='utf-8') as f:
            json.dump(self.setup, f, indent=4, ensure_ascii=False, default=str)
            
        print(f'\nSetup saved in: {setup_path}')

    def get_datamodule(self, fold_id):

        if self.train_transforms is None or self.val_transforms is None:
            train_transforms, val_transforms = self.get_transforms()
            self.train_transforms = train_transforms
            self.val_transforms = val_transforms
    
        print(f'\nDataset: {self.dataset_name}')
        print(f'Target label: {self.target_label}')
        print(f'DataModule: {self.datamodule}')
        print(f'Transforms: {self.transforms_set}')
    
        if self.datamodule == 'Generic3dDataModule':
            return Generic3dDataModule(
                dataset_path=self.dataset_path,
                fold_id=fold_id,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                train_transforms=self.train_transforms,
                val_transforms=self.val_transforms
            )
        elif self.datamodule == 'PatchCropDataModule':
            return PatchCropDataModule(
                dataset_path=self.dataset_path,
                fold_id=fold_id,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                train_transforms=self.train_transforms,
                val_transforms=self.val_transforms
            )
        elif self.datamodule == 'PatchSubsamplingDataModule':
            return PatchSubsamplingDataModule(
                dataset_path=self.dataset_path,
                fold_id=fold_id,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                train_transforms=self.train_transforms,
                val_transforms=self.val_transforms,
                resize_to=self.img_size
            )
        else:
            raise ValueError(f'Unknown datamodule: {self.datamodule}')

    @staticmethod
    def _get_dataset_labels(dataset_path):
    
        txt_path = os.path.join(dataset_path, 'valid_labels.txt')
        label_components = {}
        
        with open(txt_path, 'r') as file:
            for line in file:
                
                line = line.strip().lstrip('#')

                if not line:
                    continue
                    
                name, mapping = line.split(': ')         
                source_labels = mapping.split('->')[0]   
                
                label_components[name] = [int(x) for x in source_labels.split('|')]
                
        return label_components

    def _unique_elements(self, dict):
        return sum(1 for value in dict.values() if len(value) == 1)

    def get_transforms(self):

        if self.train_transforms is None:
            
            if self.transforms_set == 'Generic3dTransforms':
                transforms = Generic3dTransforms(self.target_label, self.labels_list, self.n_labels, self.hu_range)
                self.train_transforms, self.val_transforms = transforms.get_transforms()
            elif self.transforms_set == 'PatchCropTransforms':
                transforms = PatchCropTransforms(self.seed, self.target_label, self.labels_list, self.n_labels, self.hu_range)
                self.train_transforms, self.val_transforms = transforms.get_transforms()
            elif self.transforms_set == 'PatchSubsamplingTransforms':
                transforms = PatchSubsamplingTransforms(self.target_label, self.labels_list, self.n_labels, self.hu_range)
                self.train_transforms, self.val_transforms = transforms.get_transforms()
            else:
                raise ValueError(f'Unknown transforms set: {self.transforms_set}')
        
        return self.train_transforms, self.val_transforms
    
    def get_model_architecture(self):
    
        print(f'Architecture: {self.architecture}')
    
        if self.architecture == 'VNet':
            return VNet(
                in_channels=self.in_channels,
                num_classes=self.num_classes
            )
        elif self.architecture == 'MonaiVNet':
            return MonaiVNet(
                spatial_dims=3,
                in_channels=self.in_channels,
                out_channels=self.num_classes,
                act=("elu", {"inplace": True}),
                dropout_prob_down=0.25,
                dropout_prob_up=(0.25, 0.25),
                dropout_dim=3,
                bias=False
            )
        elif self.architecture == 'ResUNet50':
            return ResUNet50(
                in_channels=self.in_channels,
                num_classes=self.num_classes
            )
        elif self.architecture == 'HighRes3DNet':
            return HighRes3DNet(
                self.in_channels,
                self.num_classes,
                initial_out_channels_power=4,
                layers_per_residual_block=2,
                residual_blocks_per_dilation=2,
                dilations=2,
                batch_norm=False,
                instance_norm=True,
                residual=True,
                padding_mode='constant',
                add_dropout_layer=True
            )
        elif self.architecture == 'SkipDenseNet3D':
            return SkipDenseNet3D(
                    in_channels=1, 
                    num_classes=2,
                    growth_rate=16,
                    block_config=(4, 4, 4, 4),
                    num_init_features=32,
                    drop_rate=0.1,
                    bn_size=4
            )
        elif self.architecture == 'ResNetMed3D':
            return ResNetMed3D(
                in_channels=self.in_channels,                     
                num_classes=self.num_classes,                     
                layers=[3, 4, 6, 3],              
                block_inplanes=[64, 128, 256, 512],
                no_max_pool=False,
                shortcut_type='B'
            )
        elif self.architecture == 'UNETR':
            return UNETR(
                in_channels=self.in_channels,
                out_channels=self.num_classes,
                img_size=self.img_size,
                patch_size=(16, 16, 16),
                embedding_dim=768,
                depth=12,
                n_heads=12,
                mlp_ratio=4.,
                qkv_bias=False,
                p_dropout=0.,
                attn_dropout=0.
            )
        elif self.architecture == 'MonaiUNETR':
            return MonaiUNETR(
                in_channels=self.in_channels,
                out_channels=self.num_classes,
                img_size=self.img_size,
                feature_size=16,         
                hidden_size=768,           
                mlp_dim=3072,           
                num_heads=12,             
                norm_name='instance',
                conv_block=True,
                res_block=True,
                dropout_rate=0.0,
                spatial_dims=3,
                qkv_bias=False,
                save_attn=False
            )
        else:
            raise ValueError(f'Unknown architecture: {self.architecture}')
                
    def get_loss_function(self):
    
        print(f'Loss: {self.loss}')
    
        if self.loss == "DiceLoss":
            return DiceLoss(include_background=False, to_onehot_y=True, softmax=True)
        elif self.loss == "WCELoss":
            return WCELoss(n_classes=self.num_classes, clip_max=10.0)
        elif self.loss == "DiceCELoss":
            return DiceCELoss(include_background=False, to_onehot_y=True, softmax=True)
        else:
            raise ValueError(f'Unknown loss function: {self.loss}')

    def get_task(self, model_architecture, loss_function):

        print(f'Task: {self.task}\n')
    
        if self.task == "Generic3dSegmentationTask":
            return Generic3dSegmentationTask(self.setup, model_architecture, loss_function)
        elif self.task == "PatchCropSegmentationTask":
            return PatchCropSegmentationTask(self.setup, model_architecture, loss_function)
        elif self.task == "PatchSubsamplingSegmentationTask":
            return PatchSubsamplingSegmentationTask(self.setup, model_architecture, loss_function)
        else:
            raise ValueError(f'Unknown task class: {self.task}')

    def load_from_checkpoint(self, best_model_checkpoint, setup, model_architecture, loss_function):

        print(f'Task to load checkpoints: {self.task}\n')
    
        if self.task == "Generic3dSegmentationTask":
            return Generic3dSegmentationTask.load_from_checkpoint(
                    best_model_checkpoint,
                    setup=setup,
                    model=model_architecture,
                    loss_function=loss_function
                )
        elif self.task == "PatchCropSegmentationTask":
            return PatchCropSegmentationTask.load_from_checkpoint(
                    best_model_checkpoint,
                    setup=setup,
                    model=model_architecture,
                    loss_function=loss_function
                )
        elif self.task == "PatchSubsamplingSegmentationTask":
            return PatchSubsamplingSegmentationTask.load_from_checkpoint(
                    best_model_checkpoint,
                    setup=setup,
                    model=model_architecture,
                    loss_function=loss_function
                )
        else:
            raise ValueError(f'Unknown task to load checkpoints: {self.task}')

    def get_callbacks(self, fold_id):
        
        checkpoints_path =  f"model_checkpoints/{self.dataset_name}/{self.target_label}/ckpt_{self.experiment_id}/fold_{fold_id}"
        os.makedirs(checkpoints_path, exist_ok=True)

        checkpoint_best_result = ModelCheckpoint(
            dirpath=checkpoints_path,
            filename='{epoch:02d}-{val_iou:.4f}',
            monitor=self.monitor_metric,
            save_top_k=self.save_top_k,
            mode=self.monitor_mode,
            save_weights_only=True
        )

        # checkpoint_backup = ModelCheckpoint(
        #     dirpath=f"{checkpoints_path}/backups",
        #     filename='backup-{epoch:02d}',
        #     monitor='epoch',
        #     mode='max',
        #     save_top_k=1,
        #     every_n_epochs=10,
        #     save_on_train_epoch_end=True,
        #     save_weights_only=False
        # )

        progress_bar = RichProgressBar(
            leave=False
        )
        
        return [checkpoint_best_result, progress_bar]

    def get_logger(self, fold_id):
        return TensorBoardLogger(
            save_dir=self.outdir_path, 
            name=f"fold_{fold_id}"
        )

    def get_trainer(self, fold_id):

        callbacks = self.get_callbacks(fold_id)

        logger = self.get_logger(fold_id)
        
        return pl.Trainer(
            max_epochs=self.max_epochs,
            accelerator=self.accelerator,
            devices=self.devices,
            callbacks=callbacks,
            logger=logger,
            log_every_n_steps=self.log_every_n_steps,
            check_val_every_n_epoch=self.check_val_every_n_epoch,
            deterministic=False,  
            benchmark=True,
            precision='bf16-mixed',
            gradient_clip_val=1.0
        )

    def get_test_trainer(self):
        
        return pl.Trainer(
            accelerator=self.accelerator,
            devices=self.devices,
            logger=False,
            benchmark=True,
            precision='bf16-mixed'
        )

    def get_pruner_callback(self, trial):
        return PruningCallback(trial, monitor=self.monitor_metric)
        
    def get_optuna_trainer(self, trial):

        pruning_callback = self.get_pruner_callback(trial)

        progress_bar = RichProgressBar(
            leave=False
        )

        return pl.Trainer(
                max_epochs=self.max_epochs,
                accelerator=self.accelerator,
                devices=self.devices,
                callbacks=[pruning_callback, progress_bar],
                logger=False, 
                enable_checkpointing=False,
                precision='bf16-mixed',
                deterministic=False,  
                benchmark=True,
                gradient_clip_val=1.0
        )
        