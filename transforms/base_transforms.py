import numpy as np

import monai.transforms as T

class BaseTransforms:
    
    def __init__(self, target_label, labels_list, n_labels, hu_range):
        
        self.target_label = target_label
        self.labels_list = labels_list
        self.n_labels = n_labels

        self.hu_range = hu_range

    def print_valid_labels(self):
    
        print("\nvalid_labels.txt")
        for key, value in self.labels_list.items():
            print(f"{key}: {value}")

    def _prepare_labels(self):
        
        components = self.labels_list[self.target_label]
        
        orig_labels = list(range(self.n_labels))
        
        target_labels = [0] * self.n_labels
        
        for c in components:
            target_labels[c] = 1
            
        return orig_labels, target_labels
