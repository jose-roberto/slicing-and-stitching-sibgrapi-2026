from tasks.base_segmentation_task import torch, torchvision, BaseSegmentationTask

import nibabel as nib
import os

class Generic3dSegmentationTask(BaseSegmentationTask):

    def __init__(self, setup, model, loss_function):
        
        super().__init__(setup, model, loss_function)
        
    def training_step(self, batch, batch_idx):
        
        x = batch["image"]
        y = batch["mask"]
        logits = self(x)

        loss = self.loss_function(logits, y.long())

        self.log('Loss/train', loss, on_step=False, on_epoch=True, prog_bar=True, logger=False)

        if self.logger is not None:
            self.logger.experiment.add_scalar('Loss/train', loss, global_step=self.current_epoch)

        preds_one_hot = self.process_pred(logits.detach())
      
        ground_truths_one_hot = self.process_ground_truth(y)

        self.train_iou(y_pred=preds_one_hot, y=ground_truths_one_hot)
        self.train_dice(y_pred=preds_one_hot, y=ground_truths_one_hot)

        return loss

    def validation_step(self, batch, batch_idx):
        
        x = batch["image"]
        y = batch["mask"]

        with torch.no_grad():
            logits = self(x)
            loss = self.loss_function(logits, y.long())

        self.log('Loss/val', loss, on_step=False, on_epoch=True, prog_bar=True, logger=False)
        self.log('val_loss', loss, on_step=False, on_epoch=True, enable_graph=False, logger=False)

        if self.logger is not None:
            self.logger.experiment.add_scalar('Loss/val', loss, global_step=self.current_epoch)

        preds_one_hot = self.process_pred(logits)
        
        ground_truths_one_hot = self.process_ground_truth(y)

        self.val_iou(y_pred=preds_one_hot, y=ground_truths_one_hot)
        self.val_dice(y_pred=preds_one_hot, y=ground_truths_one_hot)
        self.val_confusion_matrix(y_pred=preds_one_hot, y=ground_truths_one_hot)

        if self.logger is not None and batch_idx == 0:

            writer = self.logger.experiment

            num_imgs = min(x.shape[0], self.num_imgs_to_plot)

            pixels_per_slice = (y[:num_imgs] > 0).sum(dim=(1, 3, 4))

            max_pixels, best_slices = torch.max(pixels_per_slice, dim=1)

            best_slices = torch.where(max_pixels > 0, best_slices, x.shape[2] // 2)

            logits = logits.detach().cpu()
            predictions = torch.argmax(logits, dim=1, keepdim=True)

            all_slices_for_grid = []
            
            for i in range(num_imgs):

                idx = best_slices[i].item() 

                all_slices_for_grid.append(x[i, :, idx, :, :].cpu().float())
                all_slices_for_grid.append(y[i, :, idx, :, :].cpu().float())
                all_slices_for_grid.append(predictions[i, :, idx, :, :].float())

            comparative_grid = torchvision.utils.make_grid(
                all_slices_for_grid, 
                nrow=3, 
                padding=4, 
                normalize=True 
            )
            
            writer.add_image('Input-Label-Prediction', comparative_grid, global_step=self.current_epoch)

    def test_step(self, batch, batch_idx):
         
        x = batch["image"]
        y = batch["mask"]

        logits = self(x)

        preds_one_hot = self.process_pred(logits)
        
        ground_truths_one_hot = self.process_ground_truth(y)

        self.test_iou(y_pred=preds_one_hot, y=ground_truths_one_hot)
        self.test_dice(y_pred=preds_one_hot, y=ground_truths_one_hot)
        self.test_confusion_matrix(y_pred=preds_one_hot, y=ground_truths_one_hot)

        self.test_hd(y_pred=preds_one_hot, y=ground_truths_one_hot)
        self.test_hd95(y_pred=preds_one_hot, y=ground_truths_one_hot)
        self.test_sd(y_pred=preds_one_hot, y=ground_truths_one_hot)
        
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        
        x = batch["image"]
        y = batch["mask"]

        logits = self(x)

        pred = self.process_pred(logits).detach().cpu().numpy().squeeze()

        if pred.ndim == 4: 
            pred = pred[1]

        file_path = x.meta["filename_or_obj"]
        
        if isinstance(file_path, list):
            orig_filepath = file_path[0]

        orig_nifti = nib.load(orig_filepath)
        real_affine = orig_nifti.affine

        original_filename = os.path.basename(orig_filepath)
        instance_id = f"pred_{original_filename}"
                
        file_path = os.path.join(self.prediction_output_path, instance_id)
        pred_nib = nib.Nifti1Image(pred, real_affine)
        nib.save(pred_nib, file_path)
        