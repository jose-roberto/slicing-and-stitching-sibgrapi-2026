from tasks.base_segmentation_task import torch, torchvision, AsDiscrete, BaseSegmentationTask

import nibabel as nib
import numpy as np
import os
from skimage import transform

def sticthing_tst_with_logits(net, inps_list, labs_list, off_list, size_list, strides, orig_shape, loss_fn=None):

    orig_shape = (orig_shape[0].item(),
                  orig_shape[1].item(),
                  orig_shape[2].item())

    inps_full = np.zeros(orig_shape, dtype=np.float32)
    labs_full = np.zeros(orig_shape, dtype=np.uint8)
    prds_full = np.zeros(orig_shape, dtype=np.uint8)

    # Variáveis para acompanhar a loss
    total_loss = 0.0
    num_patches = len(inps_list)

    for inps, labs, off, sz in zip(inps_list, labs_list, off_list, size_list):

        sz = (sz[0].item(), sz[1].item(), sz[2].item())

        # Casting tensors to cuda.
        device = next(net.parameters()).device 

        inps = inps.to(device)
        labs = labs.to(device)

        # Forwarding
        outs = net(inps)

        if loss_fn is not None:
            label_for_loss = labs.long()
            while label_for_loss.ndim < 5:
                label_for_loss = label_for_loss.unsqueeze(0 if label_for_loss.ndim == 3 else 1)
        
            patch_loss = loss_fn(outs, label_for_loss)
            total_loss += patch_loss.item() 

        # Obtaining predictions.
        prds = outs.detach().max(1)[1].squeeze().cpu().numpy()

        # Transforming to ndarray.
        inps = inps.detach().squeeze().cpu().numpy()
        labs = labs.detach().squeeze().cpu().numpy()

        # Reconstructing full sample, mask and prediction.
        inps = transform.resize(inps, sz, order=1, preserve_range=True, anti_aliasing=False)
        labs = transform.resize(labs, sz, order=0, preserve_range=True, anti_aliasing=False).astype(np.uint8)
        prds = transform.resize(prds, sz, order=0, preserve_range=True, anti_aliasing=False).astype(np.uint8)

        inps_full[off[0]::strides[0],
                  off[1]::strides[1],
                  off[2]::strides[2]] = inps
        labs_full[off[0]::strides[0],
                  off[1]::strides[1],
                  off[2]::strides[2]] = labs
        prds_full[off[0]::strides[0],
                  off[1]::strides[1],
                  off[2]::strides[2]] = prds
        
    avg_val_loss = (total_loss / num_patches) if num_patches > 0 else 0.0
        
    return inps_full, labs_full, prds_full, avg_val_loss
    
class PatchSubsamplingSegmentationTask(BaseSegmentationTask):

    def __init__(self, setup, model, loss_function):
        
        super().__init__(setup, model, loss_function)

        self.prediction_output_path = None

        self.process_val_pred = AsDiscrete(to_onehot=self.num_classes, dim=1)
        
    def training_step(self, batch, batch_idx):

        x = batch["image"]
        y = batch["mask"]

        logits = self(x)

        label_for_loss = y.long()
        while label_for_loss.ndim < 5:
            label_for_loss = label_for_loss.unsqueeze(0 if label_for_loss.ndim == 3 else 1)
            
        loss = self.loss_function(logits, label_for_loss)

        self.log('Loss/train', loss, on_step=False, on_epoch=True, prog_bar=True, logger=False)

        if self.logger is not None:
            self.logger.experiment.add_scalar('Loss/train', loss, global_step=self.current_epoch)

        preds_one_hot = self.process_pred(logits.detach())
        y_for_metrics = y.unsqueeze(1)
        ground_truths_one_hot = self.process_ground_truth(y_for_metrics)

        self.train_iou(y_pred=preds_one_hot, y=ground_truths_one_hot)
        self.train_dice(y_pred=preds_one_hot, y=ground_truths_one_hot)

        return loss

    def validation_step(self, batch, batch_idx):

        inps_full, labs_full, prds_full, val_loss = sticthing_tst_with_logits(
            net=self.model,
            inps_list=batch["image_list"],
            labs_list=batch["mask_list"],
            off_list=batch["off_list"],
            size_list=batch["size_list"],
            strides=batch["strides"],
            orig_shape=batch["orig_shape"],
            loss_fn=self.loss_function 
        )

        self.log('Loss/val', val_loss, on_step=False, on_epoch=True, prog_bar=True, logger=False)
        if self.logger is not None:
            self.logger.experiment.add_scalar('Loss/val', val_loss, global_step=self.current_epoch)

        x_val = torch.from_numpy(inps_full).unsqueeze(0).unsqueeze(0).to(self.device, dtype=torch.float32)
        y_val = torch.from_numpy(labs_full).unsqueeze(0).unsqueeze(0).to(self.device, dtype=torch.long)
        preds_val = torch.from_numpy(prds_full).unsqueeze(0).unsqueeze(0).to(self.device, dtype=torch.long)

        preds_one_hot = self.process_val_pred(preds_val)
        ground_truths_one_hot = self.process_ground_truth(y_val)

        self.val_iou(y_pred=preds_one_hot, y=ground_truths_one_hot)
        self.val_dice(y_pred=preds_one_hot, y=ground_truths_one_hot)
        self.val_confusion_matrix(y_pred=preds_one_hot, y=ground_truths_one_hot)

        if self.logger is not None and batch_idx == 0:

            writer = self.logger.experiment

            pixels_per_slice = (y_val[0, 0] > 0).sum(dim=(1, 2)) 

            max_pixels, best_slice_idx = torch.max(pixels_per_slice, dim=0)

            if max_pixels == 0:
                best_slice_idx = x_val.shape[2] // 2 
            else:
                best_slice_idx = best_slice_idx.item()

            all_slices_for_grid = [
                x_val[0, :, best_slice_idx, :, :].cpu().float(),
                y_val[0, :, best_slice_idx, :, :].cpu().float(),
                preds_val[0, :, best_slice_idx, :, :].cpu().float()
            ]

            comparative_grid = torchvision.utils.make_grid(
                all_slices_for_grid, 
                nrow=3, 
                padding=4, 
                normalize=True 
            ) 
                        
            writer.add_image('Input-Label-Prediction', comparative_grid, global_step=self.current_epoch)

    def test_step(self, batch, batch_idx):
        
        inps_full, labs_full, prds_full, val_loss = sticthing_tst_with_logits(
            net=self.model,
            inps_list=batch["image_list"],
            labs_list=batch["mask_list"],
            off_list=batch["off_list"],
            size_list=batch["size_list"],
            strides=batch["strides"],
            orig_shape=batch["orig_shape"],
            loss_fn=None
        )

        x_val = torch.from_numpy(inps_full).unsqueeze(0).unsqueeze(0).to(self.device, dtype=torch.float32)
        y_val = torch.from_numpy(labs_full).unsqueeze(0).unsqueeze(0).to(self.device, dtype=torch.long)
        preds_val = torch.from_numpy(prds_full).unsqueeze(0).unsqueeze(0).to(self.device, dtype=torch.long)

        preds_one_hot = self.process_val_pred(preds_val)
        ground_truths_one_hot = self.process_ground_truth(y_val)

        self.test_iou(y_pred=preds_one_hot, y=ground_truths_one_hot)
        self.test_dice(y_pred=preds_one_hot, y=ground_truths_one_hot)
        self.test_confusion_matrix(y_pred=preds_one_hot, y=ground_truths_one_hot)

        self.test_hd(y_pred=preds_one_hot, y=ground_truths_one_hot)
        self.test_hd95(y_pred=preds_one_hot, y=ground_truths_one_hot)
        self.test_sd(y_pred=preds_one_hot, y=ground_truths_one_hot)

    @torch.no_grad()
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        
        inps_full, labs_full, prds_full, val_loss = sticthing_tst_with_logits(
            net=self.model,
            inps_list=batch["image_list"],
            labs_list=batch["mask_list"],
            off_list=batch["off_list"],
            size_list=batch["size_list"],
            strides=batch["strides"],
            orig_shape=batch["orig_shape"],
            loss_fn=None
        )

        orig_filepath = batch['instance_path'][0]
        
        orig_nifti = nib.load(orig_filepath)
        real_affine = orig_nifti.affine

        original_filename = os.path.basename(orig_filepath)
        instance_id = f"pred_{original_filename}"
                
        file_path = os.path.join(self.prediction_output_path, instance_id)
        pred_nib = nib.Nifti1Image(prds_full, real_affine)
        nib.save(pred_nib, file_path)
