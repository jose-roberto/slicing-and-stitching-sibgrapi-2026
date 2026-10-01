import torch
import torch.nn as nn
import torch.nn.functional as F

def get_weights(targets, n_classes, clip_max=0.0):
   
    #targets: tensor [B, D, H, W]

    bins, counts = torch.unique(targets, return_counts=True)

    weight = torch.zeros(n_classes, device=targets.device)

    if len(counts) > 0:
        
        max_count = counts.max().float()

        for i in range(n_classes):
            if i in bins:
                idx = (bins == i).nonzero(as_tuple=True)[0]
                weight[i] = max_count / counts[idx].float()
            else:
                weight[i] = 0.0

    if clip_max > 0.0:
        weight = torch.clamp(weight, 0.0, clip_max)

    return weight


class WCELoss(nn.Module):
    
    def __init__(self, n_classes, clip_max=0.0):
        
        super().__init__()
        
        self.n_classes = n_classes
        self.clip_max = clip_max

    def forward(self, logits, targets):

        if torch.isnan(logits).any():
            print("Logits already NaN before loss")

        if targets.ndim == 5 and targets.shape[1] == 1:
            targets = targets.squeeze(1)
    
        targets = targets.long()
    
        weights = get_weights(targets, n_classes=self.n_classes, clip_max=self.clip_max)
    
        wce = F.cross_entropy(logits, targets, weight=weights)
    
        loss = wce

        if torch.isnan(loss):
            print("NaN detected")
        
            print("logits max:", logits.max())
            print("logits min:", logits.min())
        
            print("targets unique:", torch.unique(targets))
        
            print("weights:", weights)
        
            raise RuntimeError("Loss became NaN")
    
        return loss
