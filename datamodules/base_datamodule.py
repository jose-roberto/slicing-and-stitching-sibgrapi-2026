import os

import pytorch_lightning as pl

from monai.data import CacheDataset, DataLoader

class BaseDataModule(pl.LightningDataModule):
    
    def __init__(self, dataset_path, fold_id, batch_size, num_workers, train_transforms, val_transforms):
        
        super().__init__()
        
        self.dataset_path = dataset_path
        self.fold_id = fold_id
     
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        self.train_transforms = train_transforms
        self.val_transforms = val_transforms

    def _get_files(self, mode):

        image_folder = os.path.join(self.dataset_path, 'images')
        mask_folder = os.path.join(self.dataset_path, 'ground_truths')
        folds_folder = os.path.join(self.dataset_path, 'folds')

        image_paths = sorted([os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.endswith(".nii.gz")])
        mask_paths = sorted([os.path.join(mask_folder, f) for f in os.listdir(mask_folder) if f.endswith(".nii.gz")])
        
        prefix = 'trn' if mode == 'train' else 'tst'
        fold_path = os.path.join(folds_folder, f'{prefix}_f{self.fold_id}.txt')
        
        with open(fold_path) as file:
            ids = [line.strip() for line in file if line.strip()]
        
        img_paths = [p for p in image_paths if os.path.basename(p) in ids]
        msk_paths = [p for p in mask_paths if os.path.basename(p) in ids]
        
        assert len(img_paths) == len(msk_paths), f"Inconsistência de pares no fold {self.fold_id} ({mode})"
        
        return [{"image": img, "mask": msk} for img, msk in zip(img_paths, msk_paths)]

    def setup(self, stage='none'):
    
        train_files = self._get_files(mode='train')
        val_files = self._get_files(mode='val')
        
        self.train_dataset = CacheDataset(
            data=train_files,
            transform=self.train_transforms,
            cache_rate=1.0, 
            num_workers=self.num_workers 
        )
        
        self.val_dataset = CacheDataset(
            data=val_files,
            transform=self.val_transforms,
            cache_rate=1.0,
            num_workers=self.num_workers
        )
        