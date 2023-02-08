from scripts.constants import DISENTANGLER_CLASS_OUTPUT
from torch import nn
import torch

class AdversarialLoss(nn.Module):
    def __init__(self, weight=1.0, beta=1.0):
        super().__init__()
        
        self.weight = weight
        self.beta = beta
        

    def forward(self, codebook_mapping_layers, zq_nonphylo):
        nonattr_mapping_detached = codebook_mapping_layers(zq_nonphylo.detach())
        
        with torch.no_grad():
            nonattr_learning_detached = codebook_mapping_layers(zq_nonphylo)
        
        return nonattr_mapping_detached, nonattr_learning_detached


        
