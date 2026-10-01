from transforms.base_transforms import np, T, BaseTransforms

class Generic3dTransforms(BaseTransforms):
    
    def __init__(self, target_label, labels_list, n_labels, hu_range):
        
        super().__init__(target_label, labels_list, n_labels, hu_range)
        
    def get_transforms(self):
        
        orig_labels, target_labels = self._prepare_labels()

        a_min = self.hu_range[0]
        a_max = self.hu_range[1]
        
        radians = 10 * (np.pi / 180)

        self.train_transforms = T.Compose([
            T.LoadImaged(keys=['image', 'mask']),
            T.EnsureChannelFirstd(keys=['image', 'mask']),
            
            T.MapLabelValued(
                keys=["mask"],
                orig_labels=orig_labels,
                target_labels=target_labels
            ),
    
            T.ScaleIntensityRanged(
                keys=['image'],
                a_min=a_min, a_max=a_max,
                b_min=0.0, b_max=1.0,
                clip=True
            ), 
    
            T.RandFlipd(keys=["image", "mask"], spatial_axis=0, prob=0.5),
            T.RandFlipd(keys=["image", "mask"], spatial_axis=1, prob=0.5),
            T.RandFlipd(keys=["image", "mask"], spatial_axis=2, prob=0.5),
        
            T.RandAffined(
                keys=["image", "mask"],
                prob=0.5, 
                rotate_range=(radians, radians, radians),
                scale_range=(0.1, 0.1, 0.1), 
                mode=("bilinear", "nearest"),
                padding_mode="zeros"
            ),
     
            T.ToTensord(keys=['image', 'mask'])
        ])
        
        self.val_transforms = T.Compose([
            T.LoadImaged(keys=['image', 'mask']),
            T.EnsureChannelFirstd(keys=['image', 'mask']),
            
            T.MapLabelValued(
                keys=["mask"],
                orig_labels=orig_labels,
                target_labels=target_labels
            ),
            
            T.ScaleIntensityRanged(
                keys=['image'],
                a_min=a_min, a_max=a_max,
                b_min=0.0, b_max=1.0,
                clip=True
            ),
                    
            T.ToTensord(keys=['image', 'mask'])
        ])

        return self.train_transforms, self.val_transforms
        