#based on https://github.com/CompVis/taming-transformers

import torch
from scripts.analysis_utils import get_phylomapper_from_config
from scripts.import_utils import instantiate_from_config


class LabelCond(object):
    def __init__(self, phylogenyconfig=None, phyloDistances_string=None, level=3):
        self.phylo_mapper = None
        if phylogenyconfig is not None:
            self.phylo_mapper = get_phylomapper_from_config(instantiate_from_config(phylogenyconfig), phyloDistances_string, level) 
                
            
        self.level = level

    def eval(self):
        return self
    
    def encode(self, c):
        if len(c.shape) > 1:
            c_= c[:, 0]
        else:
            c_=c
            
        if self.phylo_mapper is not None:
            c_ = self.phylo_mapper.get_mapped_truth(c_)
            
        
        if len(c.shape) == 1:
            c = c_.view(c_.shape[0], 1)
        else:
            c[:, 0] = c_
            

        c_ind = c.to(dtype=torch.long)

        info = None, None, c_ind
        return c, None, info

    def decode(self, c):
        c = c[:, 0].view(c.shape[0])
        return c
