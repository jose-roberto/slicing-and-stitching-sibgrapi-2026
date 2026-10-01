import os
import numpy as np
import torch
import nibabel as nib

from torch.utils import data
import torch.nn.functional as F

from skimage import io
from skimage import transform

from scipy import ndimage as ndi

import SimpleITK as sitk

################################################################
# Constants. ###################################################
################################################################
root = '/home/oliveirahugo/scratch/Datasets/'

################################################################
# Utilitary functions. #########################################
################################################################
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
    before_crop_size = (resize_to[0] + (resize_to[0] // 16),
                        resize_to[1] + (resize_to[1] // 16),
                        resize_to[2] + (resize_to[2] // 16))
    
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

def sticthing_tst(net, inps_list, labs_list, off_list, size_list, strides, orig_shape):

    orig_shape = (orig_shape[0].item(),
                  orig_shape[1].item(),
                  orig_shape[2].item())

    inps_full = np.zeros(orig_shape, dtype=np.float32)
    labs_full = np.zeros(orig_shape, dtype=np.uint8)
    prds_full = np.zeros(orig_shape, dtype=np.uint8)

    for inps, labs, off, sz in zip(inps_list, labs_list, off_list, size_list):

        sz = (sz[0].item(),
              sz[1].item(),
              sz[2].item())

        # Casting tensors to cuda.
        inps = inps.cuda()
        labs = labs.cuda()

        # Forwarding.
        outs = net(inps)

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
        
    return inps_full, labs_full, prds_full

def load_sample(img_path, msk_path):
    
    if (img_path.endswith('.nii') and msk_path.endswith('.nii')) or (img_path.endswith('.nii.gz') and msk_path.endswith('.nii.gz')):
        
        img = nib.load(img_path)
        msk = nib.load(msk_path)
        
        # Aligning images and labels.
        msk.set_qform(img.get_qform(), code=1)
        msk.set_sform(img.get_sform(), code=0)
        
        # Getting data from nifti.
        img = img.get_fdata()
        msk = msk.get_fdata()
        
    elif img_path.endswith('.mhd') and msk_path.endswith('.mhd'):
        
        # Reads the image and label using SimpleITK.
        itkimg = sitk.ReadImage(img_path)
        itkmsk = sitk.ReadImage(msk_path)

        # Convert the image to a  numpy array first and then shuffle the dimensions to get axis in the order z,y,x
        img = sitk.GetArrayFromImage(itkimg)
        msk = sitk.GetArrayFromImage(itkmsk)
    
    return img, msk

def shift_labels(msk, valid_labels, target_labels):
    
    assert len(valid_labels) == len(target_labels)
    
    new_msk = np.zeros_like(msk)
    
    for val, t in zip(valid_labels, target_labels):
        for v in val:
            print('%d -> %d' % (v, t))
            new_msk[msk == v] = t
    
    return new_msk

################################################################
# Dataset class. ###############################################
################################################################
class ListDataset(data.Dataset):
    
    def __init__(self, mode, dataset, task, fold, resize_to):
        
        # Initializing variables.
        self.mode = mode
        self.dataset = dataset
        self.task = task
        self.fold = fold
        self.resize_to = resize_to
        
        self.num_classes = 0
        self.valid_labels = []
        self.valid_label_names = []
        self.weights = None
        
        # Creating list of paths.
        self.imgs = self.make_dataset()
        
        # Check for consistency in list.
        if len(self.imgs) == 0:
            
            raise (RuntimeError('Found 0 images, please check the data set'))

    def make_dataset(self):
        
        # Making sure the mode is correct.
        assert self.mode in ['train', 'test']
        items = []
        
        # Setting string for the mode.
        mode_str = None
        if self.mode == 'train':
            mode_str = 'trn'
        elif self.mode == 'test':
            mode_str = 'tst'
            
        # Joining input paths.
        img_dir = os.path.join(root, self.dataset, 'images')
        msk_dir = os.path.join(root, self.dataset, 'ground_truths', self.task)
        valid_label_path = os.path.join(root, self.dataset, '%s_valid_labels.txt' % (self.task))
        weight_path = os.path.join(root, self.dataset, '%s_weights.txt' % (self.task))
        
        # Reading valid labels from txt.
        with open(valid_label_path, 'r') as valid_label_file:
            lines = [l for l in valid_label_file.readlines() if l != '']
            self.valid_labels = [[int(v) for v in l.split(': ')[-1].split('->')[0].split('|')] if '|' in l.split(': ')[-1].split('->')[0] else [int(l.split(': ')[-1].split('->')[0])] for l in lines]
            self.target_labels = [int(l.split(': ')[-1].split('->')[1]) for l in lines]
            self.valid_label_names = [l.split(': ')[0] for l in lines]
            self.num_classes = len(self.valid_labels)
        
        # Reading class weights for loss calculation from txt.
        try:
            with open(weight_path, 'r') as weight_file:
                self.weights = [float(l) for l in weight_file.readlines() if l != '']
        except:
            self.weights = [1.0 for c in range(self.num_classes)]
        
        # Reading sample file names from text file.
        data_list = [l.strip('\n') for l in open(os.path.join(root, self.dataset, self.task + '_' + mode_str + '_f' + str(self.fold) + '.txt')).readlines()]
        
        # Creating list containing image and ground truth paths.
        for it in data_list:
            item = (os.path.join(img_dir, it), os.path.join(msk_dir, it))
            items.append(item)
            
        # Returning list.
        return items
    
    def __getitem__(self, index):
        
        # Reading items from list.
        img_path, msk_path = self.imgs[index]
        
        # Reading image and label.
        img, msk = load_sample(img_path, msk_path)
        
        # Sorting and shifting labels in mask if necessary.
        if self.valid_labels != [c for c in range(self.num_classes)]:
            msk = shift_labels(msk, self.valid_labels, self.target_labels)
        
        # Normalization.
        img = (img - img.mean()) / (img.std() + 1e-7)
        
        # Train and test transformations.
        if self.mode == 'train':
            
            # Slicing for train.
            img, msk = slicing_trn(img, msk, self.resize_to)
            
            # Casting image and mask to the appropriate dtypes.
            img = img.astype(np.float32)
            msk = msk.astype(np.int64)
            
            # Adding channel dimension.
            img = np.expand_dims(img, axis=0)
            
            # Turning to tensors.
            img = torch.from_numpy(img)
            msk = torch.from_numpy(msk)
            
            # Returning to iterator.
            return img, msk, img_path.split('/')[-1]
        
        elif self.mode == 'test':
            
            # Saving original size.
            orig_shape = img.shape
            
            # Slicing.
            img, msk, off, size, strides = slicing_tst(img, msk, self.resize_to)
            
            # Returning to iterator.
            return img, msk, off, size, strides, orig_shape, img_path.split('/')[-1]

    def __len__(self):

        return len(self.imgs)
