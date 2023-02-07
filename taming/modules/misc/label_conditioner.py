import random
from taming.analysis_utils import get_phylomapper_from_config
from taming.import_utils import instantiate_from_config
import torch

class LabelCond(object):
    def __init__(self, postfix_codes=False, partial_codes=False, level_codes=False, phylogenyconfig=None, phyloDistances_string=None, anti_class=False, level=3):
        self.phylo_mapper = None
        self.anti_phylo_mapper = None
        if phylogenyconfig is not None:
            self.phylo_mapper = get_phylomapper_from_config(instantiate_from_config(phylogenyconfig), phyloDistances_string, level) # 
            if anti_class:
                self.anti_phylo_mapper = get_phylomapper_from_config(instantiate_from_config(phylogenyconfig), phyloDistances_string, level-1)
                
            
        self.postfix_codes = postfix_codes
        self.partial_codes = partial_codes
        self.level_codes = level_codes
        self.level = level
        self.anti_class = anti_class

    def eval(self):
        return self

    def get_siblings(self, x):
        siblings = self.anti_phylo_mapper.sibling_mapper(x,  self.anti_phylo_mapper.outputname)
        if len(siblings) > 1:
            siblings.remove(x.item()) 
        choice = random.choice(siblings)
        return choice
    
    def encode(self, c):
        if len(c.shape) > 1:
            c_= c[:, 0]
        else:
            c_=c
            
        if not self.anti_class:
            if self.phylo_mapper is not None:
                c_ = self.phylo_mapper.get_mapped_truth(c_)
        else:
            if self.anti_phylo_mapper is None:
                raise "phylo_mapper is needed for anti_class"
            if self.level==0:
                raise "anti_class is not defined for phylogeny level 0"            
            
            c_ = torch.LongTensor(list(map(lambda x: self.get_siblings(x), c_))).to(c_.device) 
            
        
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
