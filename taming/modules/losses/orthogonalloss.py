from turtle import forward
from torch import nn
import torch
import numpy as np

class OrthogonalLoss(nn.Module):
    def __init__(self, weight, stride = 1, padding = 0, verbose=False):
        super().__init__()

        self.weight=weight
        self.stride=stride
        self.padding=padding

    # def forward(self, cumulative_loss, z1, z2):
    #     losses_dict = {}
        
    #     z1_flattened = z1.reshape(z1.shape[0], -1)
    #     z2_flattened = z2.reshape(z2.shape[0], -1)
    #     print(z1_flattened.shape, z2_flattened.shape)
    #     orthogonality_loss = torch.sum(z1_flattened * z2_flattened, dim = -1)

    #     losses_dict['cumulative_loss'] = cumulative_loss + self.weight*orthogonality_loss

    def forward(self, kernel):
        [o_c, i_c, w, h] = kernel.shape
        # print(kernel.shape)
        output = torch.conv2d(kernel, kernel, stride=self.stride, padding=self.padding)
        # print(output.shape)
        target = torch.zeros((o_c, o_c, output.shape[-2], output.shape[-1])).to(output.device)
        ct = int(np.floor(output.shape[-1]/2))
        target[:,:,ct,ct] = torch.eye(o_c).to(output.device)
        return torch.norm(output - target)

        
