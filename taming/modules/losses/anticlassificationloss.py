from torch import nn
import torch

class AntiClassificationLoss(nn.Module):
    def __init__(self, weight=1.0, beta=1.0, verbose=False):
        super().__init__()
        
        self.weight = weight
        self.beta = beta
        

    def forward(self, codebook_mapping_layers, zq_nonphylo, zq_phylo):
        # outputs[CONSTANTS.DISENTANGLER_NON_ATTRIBUTE_TO_ATTRIBUTE_OUTPUT] = self.codebook_mapping_layers(zq_nonphylo.detach())
        # for param in codebook_mapping_layers.parameters():
        #     param.requires_grad = True
        nonattr_mapping_detached = codebook_mapping_layers(zq_nonphylo.detach())
        
        # for param in codebook_mapping_layers.parameters():
        #     param.requires_grad = False
        with torch.no_grad():
            nonattr_learning_detached = codebook_mapping_layers(zq_nonphylo)
        
        anti_classification_mapping_loss = torch.abs(zq_phylo.detach().contiguous() - nonattr_mapping_detached.contiguous())
        anti_classification_learning_loss = -torch.abs(zq_phylo.detach().contiguous() - nonattr_learning_detached.contiguous())
        
        return torch.mean(anti_classification_mapping_loss),  torch.mean(anti_classification_learning_loss)


        
