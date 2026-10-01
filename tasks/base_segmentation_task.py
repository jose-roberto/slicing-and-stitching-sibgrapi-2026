import torch
import torchvision

import pytorch_lightning as pl

from monai.transforms import AsDiscrete
from monai.metrics import (
    MeanIoU,
    DiceMetric,
    ConfusionMatrixMetric,
    HausdorffDistanceMetric,
    SurfaceDistanceMetric
)
                
class BaseSegmentationTask(pl.LightningModule):

    def __init__(self, setup, model, loss_function):
        
        super().__init__()

        self.setup = setup
        
        self.model = model

        self.loss_function = loss_function
        
        self.num_classes = setup['num_classes']

        self.one_cycle_lr = setup['one_cycle_lr']
        self.base_lr = setup['base_lr']
        self.weight_decay = setup['weight_decay']
        self.max_epochs = setup['max_epochs']
        self.max_lr = setup['max_lr']
        self.pct_start = setup['pct_start']
        self.div_factor = setup['div_factor']
        self.final_div_factor = setup['final_div_factor']
        self.monitor_metric = setup['monitor_metric']
        self.monitor_mode = setup['monitor_mode'] 

        self.num_imgs_to_plot = setup['num_imgs_to_plot']
        
        self.process_pred = AsDiscrete(argmax=True, to_onehot=self.num_classes, dim=1)
        self.process_ground_truth = AsDiscrete(to_onehot=self.num_classes, dim=1)

        self.train_iou = MeanIoU(include_background=False, reduction='mean')
        self.train_dice = DiceMetric(include_background=False, reduction='mean')
        
        self.val_dice = DiceMetric(include_background=False, reduction='mean')
        self.val_iou = MeanIoU(include_background=False, reduction='mean')

        self.val_confusion_matrix = ConfusionMatrixMetric(
            include_background=False, 
            metric_name=["precision", "recall"], 
            reduction='mean'
        )

        self.test_iou = MeanIoU(include_background=False, reduction='mean')
        self.test_dice = DiceMetric(include_background=False, reduction='mean')

        self.test_confusion_matrix = ConfusionMatrixMetric(
            include_background=False, 
            metric_name=["precision", "recall"], 
            reduction='mean'
        )

        self.test_hd = HausdorffDistanceMetric(include_background=False, percentile=100.0, distance_metric='euclidean', reduction='mean', get_not_nans=True)
        self.test_hd95 = HausdorffDistanceMetric(include_background=False, percentile=95.0, distance_metric='euclidean', reduction='mean', get_not_nans=True)
        self.test_sd = SurfaceDistanceMetric(include_background=False, distance_metric='euclidean', reduction='mean', get_not_nans=True)

    def forward(self, x):
        return self.model(x)
     
    def configure_optimizers(self):
        
        optimizer = torch.optim.AdamW(
            self.parameters(), 
            lr=self.base_lr, 
            weight_decay=self.weight_decay
        )

        configs = {
            "optimizer": optimizer
        }

        if self.one_cycle_lr:
            steps_per_epoch = len(self.trainer.datamodule.train_dataloader())
    
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=self.max_lr,            
                epochs=self.max_epochs,     
                steps_per_epoch=steps_per_epoch,     
                pct_start=self.pct_start,            
                anneal_strategy='cos',
                div_factor=self.div_factor,           
                final_div_factor=self.final_div_factor        
            )

            configs["lr_scheduler"] = {
                    "scheduler": scheduler,
                    "interval": "step"
            }
    
        return configs
        
    def on_train_epoch_end(self):

        avg_train_dice = self.train_dice.aggregate().item()
        avg_train_iou = self.train_iou.aggregate().item()

        self.log('IoU/train', avg_train_iou, on_step=False, on_epoch=True, logger=False)
        self.log('Dice/train', avg_train_dice, on_step=False, on_epoch=True, logger=False)

        if self.logger is not None:
            self.logger.experiment.add_scalar('IoU/train', avg_train_iou, global_step=self.current_epoch)
            self.logger.experiment.add_scalar('Dice/train', avg_train_dice, global_step=self.current_epoch)

        self.train_dice.reset()
        self.train_iou.reset()

    def on_validation_epoch_end(self):

        avg_val_iou = self.val_iou.aggregate().item()
        avg_val_dice = self.val_dice.aggregate().item()
        
        precision_tensor, recall_tensor = self.val_confusion_matrix.aggregate()
        avg_val_precision = precision_tensor.mean().item()
        avg_val_recall = recall_tensor.mean().item()

        metrics_dict = {
            'IoU/val': avg_val_iou,
            'Dice/val': avg_val_dice,
            'Precision/val': avg_val_precision,
            'Recall/val': avg_val_recall
        }

        for name, value in metrics_dict.items():

            prog_bar = False
            if name == "IoU/val":
                prog_bar = True           
                self.log('val_iou', value, on_step=False, on_epoch=True, enable_graph=False, logger=False)
            
            self.log(name, value, on_step=False, on_epoch=True, prog_bar=prog_bar, logger=False)
            
            if self.logger is not None:
                self.logger.experiment.add_scalar(name, value, global_step=self.current_epoch)

        self.val_iou.reset()
        self.val_dice.reset()
        self.val_confusion_matrix.reset()

    def on_test_epoch_end(self):

        avg_test_iou = self.test_iou.aggregate().item()
        avg_test_dice = self.test_dice.aggregate().item()
        
        precision_tensor, recall_tensor = self.test_confusion_matrix.aggregate()
        avg_test_precision = precision_tensor.mean().item()
        avg_test_recall = recall_tensor.mean().item()

        test_hd_tensor, _ = self.test_hd.aggregate()
        avg_test_hd = test_hd_tensor.item()
        
        test_hd95_tensor, _ = self.test_hd95.aggregate()
        avg_test_hd95 = test_hd95_tensor.item()
        
        test_sd_tensor, _ = self.test_sd.aggregate()
        avg_test_sd = test_sd_tensor.item()

        metrics_dict = {
            'IoU/test': avg_test_iou,
            'Dice/test': avg_test_dice,
            'Precision/test': avg_test_precision,
            'Recall/test': avg_test_recall,
            'HausdorffDistance/test': avg_test_hd,
            'HausdorffDistance95/test': avg_test_hd95,
            'SurfaceDistance/test': avg_test_sd
        }

        self.log_dict(metrics_dict)

        self.test_iou.reset()
        self.test_dice.reset()
        self.test_confusion_matrix.reset()
        
        self.test_hd.reset()
        self.test_hd95.reset()
        self.test_sd.reset()