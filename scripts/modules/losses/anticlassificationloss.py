from scripts.constants import DISENTANGLER_CLASS_OUTPUT
from torch import nn
import torch

class AntiClassificationLoss(nn.Module):
    def __init__(self, weight=1.0, beta=1.0, v2=False, v3=False, verbose=False):
        super().__init__()
        
        self.weight = weight
        self.beta = beta
        self.v2=v2
        self.v3=v3
        

    def forward(self, codebook_mapping_layers, zq_nonphylo):#, zq_phylo): #TODO: clean this up
        # outputs[CONSTANTS.DISENTANGLER_NON_ATTRIBUTE_TO_ATTRIBUTE_OUTPUT] = self.codebook_mapping_layers(zq_nonphylo.detach())
        # for param in codebook_mapping_layers.parameters():
        #     param.requires_grad = True
        nonattr_mapping_detached = codebook_mapping_layers(zq_nonphylo.detach())
        
        # for param in codebook_mapping_layers.parameters():
        #     param.requires_grad = False
        with torch.no_grad():
            nonattr_learning_detached = codebook_mapping_layers(zq_nonphylo)
        
        # anti_classification_mapping_loss = torch.abs(zq_phylo.detach().contiguous() - nonattr_mapping_detached.contiguous())
        # anti_classification_learning_loss = -torch.abs(zq_phylo.detach().contiguous() - nonattr_learning_detached.contiguous())
        
        # return torch.mean(anti_classification_mapping_loss),  torch.mean(anti_classification_learning_loss)
        return nonattr_mapping_detached, nonattr_learning_detached


        
