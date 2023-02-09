# based on https://github.com/samaonline/Orthogonal-Convolutional-Neural-Networks

from torch import nn
import torch
import numpy as np

class OrthogonalLoss(nn.Module):
    def __init__(self, weight, stride = 1, padding = 0, verbose=False):
        super().__init__()

        self.weight=weight
        self.stride=stride
        self.padding=padding


    def forward(self, kernel):
        [o_c, i_c, w, h] = kernel.shape
        output = torch.conv2d(kernel, kernel, stride=self.stride, padding=self.padding)
        target = torch.zeros((o_c, o_c, output.shape[-2], output.shape[-1])).to(output.device)
        ct = int(np.floor(output.shape[-1]/2))
        target[:,:,ct,ct] = torch.eye(o_c).to(output.device)
        return torch.norm(output - target)

        
