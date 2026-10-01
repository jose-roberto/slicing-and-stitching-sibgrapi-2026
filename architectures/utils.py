import torch.nn as nn

def initialize_unetr_weights(*models):
    for model in models:
        for module in model.modules():
            if isinstance(module, nn.Conv3d) or isinstance(module, nn.ConvTranspose3d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='leaky_relu')
                if module.bias is not None:
                    module.bias.data.zero_()
            
            elif isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    module.bias.data.zero_()
            
            elif isinstance(module, nn.InstanceNorm3d) or isinstance(module, nn.LayerNorm):
                if getattr(module, 'weight', None) is not None:
                    module.weight.data.fill_(1)
                if getattr(module, 'bias', None) is not None:
                    module.bias.data.zero_()

####################################################################################
# Implementation from: https://github.com/hugo-oliveira/STAP-3DSegmentation ########
####################################################################################

def initialize_weights(*models):
    for model in models:
        for module in model.modules():
            if isinstance(module, nn.Conv3d) or isinstance(module, nn.Linear) or isinstance(module, nn.ConvTranspose3d):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.BatchNorm3d) or isinstance(module, nn.InstanceNorm3d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()


