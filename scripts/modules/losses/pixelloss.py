#based on https://github.com/CompVis/taming-transformers

import torch.nn as nn
import torch

def get_ce(pred: torch.Tensor, target: torch.Tensor, dim:int)-> torch.Tensor:
    out = - target * torch.log(pred.clamp(min=1e-11)) # clamp to prevent gradient explosion
    return out.sum(dim)


class PixelLoss(nn.Module):
    def __init__(self):
        super(PixelLoss, self).__init__()

    def forward(self, pred, target, reduction = True):
        loss = get_ce(pred, target, dim=1)

        if reduction:
            return loss.mean()
        else:
            return loss