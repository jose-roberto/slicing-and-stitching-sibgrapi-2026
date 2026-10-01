from datamodules.base_datamodule import os, pl, BaseDataModule, CacheDataset, DataLoader

import numpy as np
from scipy import ndimage as ndi
from skimage import transform

import torch
            
def slicing_trn(img, msk, resize_to):
    
    # Subsampling.
    y_stride = (img.shape[0] // resize_to[0]) + 1
    x_stride = (img.shape[1] // resize_to[1]) + 1
    z_stride = (img.shape[2] // resize_to[2]) + 1
    
    y_off = np.random.randint(y_stride)
    x_off = np.random.randint(x_stride)
    z_off = np.random.randint(z_stride)
    
    img = img[y_off::y_stride, x_off::x_stride, z_off::z_stride]
    msk = msk[y_off::y_stride, x_off::x_stride, z_off::z_stride]
    
    # Resizing.
    before_crop_size = (resize_to[0] + (resize_to[0] // 8),
                        resize_to[1] + (resize_to[1] // 8),
                        resize_to[2] + (resize_to[2] // 8))
    
    img = transform.resize(img, before_crop_size, order=1, preserve_range=True, anti_aliasing=False)
    msk = transform.resize(msk, before_crop_size, order=0, preserve_range=True, anti_aliasing=False)
    
    # Random crop to size resize_to.
    rnd_crop = (np.random.randint(low=0, high=(before_crop_size[0] - resize_to[0])),
                np.random.randint(low=0, high=(before_crop_size[1] - resize_to[1])),
                np.random.randint(low=0, high=(before_crop_size[2] - resize_to[2])))
    
    img = img[rnd_crop[0]:(rnd_crop[0] + resize_to[0]),
              rnd_crop[1]:(rnd_crop[1] + resize_to[1]),
              rnd_crop[2]:(rnd_crop[2] + resize_to[2])]
    msk = msk[rnd_crop[0]:(rnd_crop[0] + resize_to[0]),
              rnd_crop[1]:(rnd_crop[1] + resize_to[1]),
              rnd_crop[2]:(rnd_crop[2] + resize_to[2])]
    
    # Randomly flipping image on axis 1.
    if np.random.random() > 0.5:
        img = np.flip(img, axis=1)
        msk = np.flip(msk, axis=1)
    
    # Randomly rotating the volume for a few degrees across the axial plane.
    angle = np.random.randn()
    img = ndi.rotate(img, angle, axes=(0, 1), order=1, reshape=False)
    msk = ndi.rotate(msk, angle, axes=(0, 1), order=0, reshape=False)
    
    return img, msk

def slicing_tst(img, msk, resize_to):

    img_list = []
    msk_list = []
    off_list = []
    patch_size_list = []

    y_stride = (img.shape[0] // resize_to[0]) + 1
    x_stride = (img.shape[1] // resize_to[1]) + 1
    z_stride = (img.shape[2] // resize_to[2]) + 1

    strides = (y_stride, x_stride, z_stride)

    for y_off in range(y_stride):
        for x_off in range(x_stride):
            for z_off in range(z_stride):

                img_crop = img[y_off::y_stride,
                               x_off::x_stride,
                               z_off::z_stride]
                msk_crop = msk[y_off::y_stride,
                               x_off::x_stride,
                               z_off::z_stride]

                crop_shape = img_crop.shape
                img_crop = transform.resize(img_crop, resize_to, order=1, preserve_range=True, anti_aliasing=False)
                msk_crop = transform.resize(msk_crop, resize_to, order=0, preserve_range=True, anti_aliasing=False)

                img_crop = img_crop.astype(np.float32)
                msk_crop = msk_crop.astype(np.int64)

                img_crop = np.expand_dims(img_crop, 0)

                img_crop = torch.from_numpy(img_crop)
                msk_crop = torch.from_numpy(msk_crop)

                img_list.append(img_crop)
                msk_list.append(msk_crop)
                off_list.append((y_off, x_off, z_off))
                patch_size_list.append(crop_shape)

    return img_list, msk_list, off_list, patch_size_list, strides

class Dataset(CacheDataset):
    
    def __init__(self, data, mode, transform, resize_to, cache_rate, num_workers):

        super().__init__(data=data, transform=transform, cache_rate=cache_rate, num_workers=num_workers)

        self.mode = mode
        self.resize_to = resize_to

    def __getitem__(self, index):
        
        data = super().__getitem__(index)

        img = data['image']
        msk = data['mask']

        if self.mode == 'train':

            img, msk = slicing_trn(img, msk, self.resize_to)

            # Casting image and mask to the appropriate dtypes.
            img = img.astype(np.float32)
            msk = msk.astype(np.int64)
            
            # # Adding channel dimension.
            img = np.expand_dims(img, axis=0)
            
            # Turning to tensors.
            img = torch.from_numpy(img)
            msk = torch.from_numpy(msk)

            return {
                "image": img, "mask": msk
            }
        
        elif self.mode in ['val', 'test']:
            
            orig_shape = img.shape
            
            img_list, msk_list, off_list, size_list, strides = slicing_tst(img, msk, self.resize_to)

            instance_path = self.data[index]['image']

            return {
                "image_list": img_list,
                "mask_list": msk_list,
                "off_list": off_list,
                "size_list": size_list,
                "strides": strides,
                "orig_shape": orig_shape,
                "instance_path": instance_path
            }

class PatchSubsamplingDataModule(BaseDataModule):
    
    def __init__(self, dataset_path, fold_id, batch_size, num_workers, train_transforms, val_transforms, resize_to):
        
        super().__init__(dataset_path, fold_id, batch_size, num_workers, train_transforms, val_transforms)

        self.resize_to = resize_to

    def setup(self, stage='none'):

        if stage == 'fit' or stage is None:
            
            train_files = self._get_files(mode='train')
            val_files = self._get_files(mode='val')

            self.train_dataset = Dataset(
                data=train_files,
                mode='train',
                transform=self.train_transforms,
                resize_to=self.resize_to,
                cache_rate=1.0,
                num_workers=self.num_workers
            )
            
            self.val_dataset = Dataset(
                data=val_files, 
                mode='val',
                transform=self.val_transforms,
                resize_to=self.resize_to,
                cache_rate=1.0,
                num_workers=self.num_workers
            )

        if stage in ['test', 'predict'] or stage is None:
 
            test_files = self._get_files(mode='test') 
            
            self.test_dataset = Dataset(
                data=test_files, 
                mode='test',
                transform=self.val_transforms,
                resize_to=self.resize_to,
                cache_rate=1.0,
                num_workers=self.num_workers
            )
       
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=1, shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=1, shuffle=False, num_workers=self.num_workers)
