import torch

class LabelCond(object):
    def __init__(self, num_of_classes):
        self.num_of_classes = num_of_classes

    def eval(self):
        return self

    def encode(self, c):
        c = c.view(c.shape[0], 1)
        assert 0 <= c.min() and c.max() <= self.num_of_classes

        c_ind = c.to(dtype=torch.long)

        info = None, None, c_ind
        return c, None, info

    def decode(self, c):
        c = c.view(c.shape[0])
        return c
