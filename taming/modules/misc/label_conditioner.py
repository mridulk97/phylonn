from taming.analysis_utils import get_phylomapper_from_config
from taming.import_utils import instantiate_from_config
import torch

class LabelCond(object):
    def __init__(self, num_of_classes, phylogenyconfig=None, phyloDistances_string=None, level=3):
        # self.num_of_classes = num_of_classes
        self.phylo_mapper = None
        if phylogenyconfig is not None:
            self.phylo_mapper = get_phylomapper_from_config(instantiate_from_config(phylogenyconfig), phyloDistances_string, level)

    def eval(self):
        return self

    def encode(self, c):
        if self.phylo_mapper is not None:
            c = self.phylo_mapper.get_mapped_truth(c)
        
        c = c.view(c.shape[0], 1)

        c_ind = c.to(dtype=torch.long)

        info = None, None, c_ind
        return c, None, info

    def decode(self, c):
        c = c.view(c.shape[0])
        return c
