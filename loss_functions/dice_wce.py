import torch
import torch.nn as nn
import torch.nn.functional as F

from monai.losses import DiceLoss

def get_weights(targets, n_classes, clip_max=0.0):
    
    # targets: tensor [B, D, H, W]
    
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

class DiceWCELoss(nn.Module):
    
    def __init__(self, n_classes, clip_max=0.0, lambda_dice=1.0, lambda_ce=1.0):
        super().__init__()
        
        self.n_classes = n_classes
        self.clip_max = clip_max
        self.lambda_dice = lambda_dice
        self.lambda_ce = lambda_ce

        self.monai_dice = DiceLoss(
            to_onehot_y=True, 
            softmax=True, 
            include_background=False
        )

    def forward(self, logits, targets):
        
        if torch.isnan(logits).any():
            print("Logits already NaN before loss")

        targets_ce = targets
        if targets_ce.ndim == 5 and targets_ce.shape[1] == 1:
            targets_ce = targets_ce.squeeze(1)
        targets_ce = targets_ce.long()

        targets_dice = targets
        if targets_dice.ndim == 4:
            targets_dice = targets_dice.unsqueeze(1)

        targets_dice = targets_dice.long()

        weights = get_weights(targets_ce, n_classes=self.n_classes, clip_max=self.clip_max)
        wce_loss = F.cross_entropy(logits, targets_ce, weight=weights)

        dice_loss = self.monai_dice(logits, targets_dice)

        loss = (self.lambda_ce * wce_loss) + (self.lambda_dice * dice_loss)

        if torch.isnan(loss):
            print("NaN detected")
            print("logits max:", logits.max())
            print("logits min:", logits.min())
            print("targets unique:", torch.unique(targets_ce))
            print("weights:", weights)
            raise RuntimeError("Loss became NaN")
    
        return loss
        