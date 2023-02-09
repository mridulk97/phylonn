#based on https://github.com/CompVis/taming-transformers

import torch.nn as nn
import numpy as np


class AbstractPermuter(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
    def forward(self, x, reverse=False):
        raise NotImplementedError


class Identity(AbstractPermuter):
    def __init__(self):
        super().__init__()

    def forward(self, x, reverse=False):
        return x
