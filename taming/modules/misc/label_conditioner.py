from main import instantiate_from_config
from taming.analysis_utils import get_phylomapper_from_config
import torch

class LabelCond(object):
    def __init__(self, num_of_classes, phylogenyconfig=None, phyloDistances_string=None, level=3):
        # self.num_of_classes = num_of_classes
        self.phylo_mapper = None
        if phylogenyconfig is not None:
            self.phylo_mapper = get_phylomapper_from_config(instantiate_from_config(phylogenyconfig), phyloDistances_string, level)
            # self.phylogeny = instantiate_from_config(phylogenyconfig)
            # self.phylo_distances = parse_phyloDistances(phyloDistances_string)
            # self.siblingfinder = Species_sibling_finder(self.phylogeny, self.phylo_distances)
            # relative_distance = get_relative_distance_for_level(self.phylo_distances, level)
            # species_groups = self.phylogeny.get_species_groups(relative_distance)
            # species_groups_representatives = list(map(lambda x: x[0], species_groups))
            # self.mlb = list(map(lambda x: self.phylogeny.getLabelList().index(x), species_groups_representatives))
    

    def eval(self):
        return self

    def encode(self, c):
        # print(c)
        if self.phylo_mapper is not None:
            c = self.phylo_mapper.get_mapped_truth(c)
            # layer_truth = list(map(lambda x: self.siblingfinder.map_speciesId_siblingVector(x, str(self.phylo_distances[self.level]).replace(".", "")+"distance"), c))
            # c = torch.LongTensor(list(map(lambda x: self.mlb.index(x[0]), layer_truth))).to(c.device)
        
        c = c.view(c.shape[0], 1)
        # print(c)
        # print('***************')
        # assert 0 <= c.min() and c.max() <= self.num_of_classes

        c_ind = c.to(dtype=torch.long)

        info = None, None, c_ind
        return c, None, info

    def decode(self, c):
        c = c.view(c.shape[0])
        return c
