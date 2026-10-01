from datamodules.base_datamodule import BaseDataModule, DataLoader

class Generic3dDataModule(BaseDataModule):
    
    def __init__(self, dataset_path, fold_id, batch_size, num_workers, train_transforms, val_transforms):
        
        super().__init__(dataset_path, fold_id, batch_size, num_workers, train_transforms, val_transforms)
       
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=False, persistent_workers=False)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=1, shuffle=False, num_workers=self.num_workers, pin_memory=False, persistent_workers=False)

    def test_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=1, shuffle=False, num_workers=self.num_workers, pin_memory=False, persistent_workers=False)
