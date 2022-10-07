import torch.nn as nn
from sklearn.preprocessing import MultiLabelBinarizer
import torch
from torchmetrics import F1Score
# from torchmetrics.classification import MultilabelF1Score

from main import instantiate_from_config

def parse_phyloDistances(phyloDistances_string):
    phyloDistances_list_string = phyloDistances_string.split(",")
    sorted_distance = sorted(list(map(lambda x: float(x), phyloDistances_list_string)))
    return sorted_distance


class Species_sibling_finder():
    # Contructor
    def __init__(self, phylogeny, genetic_distances):
        self.map = {}
        self.phylogeny = phylogeny
        for species in phylogeny.node_ids:
            self.map[species] = {}
            for distance in genetic_distances:
                self.map[species][str(distance).replace(".", "")+"distance"] = phylogeny.get_siblings_by_name(species, distance)

        # print('self.map', self.map)

    def map_speciesId_siblingVector(self, speciesId, loss_name):
        label_list = self.phylogeny.getLabelList()
        species = label_list[speciesId]
        siblings = self.map[species][loss_name]
        siblings_indices = list(map(lambda x: label_list.index(x), siblings))
        # print('siblings0', loss_name, speciesId, species, siblings, siblings_indices, range(len(fine_list)))
        return siblings_indices

###----------------------------------###





class PhyloLoss(nn.Module):
    def __init__(self, phyloDistances_string, phylogenyconfig, phylo_weight, fc_layers, verbose=False):
        super().__init__()

        self.phylo_distances = parse_phyloDistances(phyloDistances_string)
        self.phylogeny = instantiate_from_config(phylogenyconfig)
        self.siblingfinder = Species_sibling_finder(self.phylogeny, self.phylo_distances)
        self.phylo_weight = phylo_weight
        self.classifier_output_size = len(self.phylogeny.getLabelList())
        self.fc_layers = fc_layers

        self.verbose = verbose

        self.criterionBCE = torch.nn.BCEWithLogitsLoss()
        self.criterionCE = torch.nn.CrossEntropyLoss() #TODO: we might want to look at tricks such as weight balancing (see get_criterion from HGNN). For now, it is fine.
        if self.verbose:
            self.F1 = F1Score(num_classes=self.classifier_output_size, multiclass=True)
            # self.F1_multilabel = MultilabelF1Score(num_labels=self.classifier_output_size) # NOTE: Does not work for torchmetrics < 0.10

    def forward(self, cumulative_loss, activations, labels):

        losses_dict = {'class_loss': self.criterionCE(activations['class'], labels)}

        for loss_name, activation in activations.items():
            if loss_name=='class':
                continue
            layer_truth = list(map(lambda x: self.siblingfinder.map_speciesId_siblingVector(x, loss_name), labels))
            mlb = MultiLabelBinarizer(classes = list(range(self.classifier_output_size)))
            hotcoded_siblingindices = torch.FloatTensor(mlb.fit_transform(layer_truth)).to(activation.device)
            losses_dict[loss_name+"_loss"] = self.criterionBCE(activation, hotcoded_siblingindices)
            # losses_dict[loss_name+"_f1"] = self.F1_multilabel(activation, hotcoded_siblingindices)

        # aggregate the losses
        # NOTE: Don't add losses you don't want to add up before here!
        total_phylo_loss = torch.stack(list(losses_dict.values())).sum()
        losses_dict['total_phylo_loss'] = total_phylo_loss
        losses_dict['cumulative_loss'] = cumulative_loss + self.phylo_weight*total_phylo_loss

        if self.verbose:
            class_f1 = self.F1(activations['class'], labels)
            losses_dict['class_f1'] = class_f1

        # return loss_dic
        return losses_dict
