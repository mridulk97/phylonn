from taming.loading_utils import load_config, load_phylovqvae
from taming.analysis_utils import DISTANCE_DICT
from taming.models.vqgan import VQModel
from taming.plotting_utils import get_fig_pth, plot_heatmap

import torch

import taming.constants as CONSTANTS
from taming.data.custom import CustomTest as CustomDataset

import os

from omegaconf import OmegaConf
import argparse

import pickle

GENERATED_FOLDER = "most_likely_generations"

#*****************

class HistogramParser_VQGAN:
    def __init__(self, model, distance_used):
        self.possible_codes = model.quantize.n_e
        self.distance_used = distance_used
    
    def get_distances(self, hist, species1, species2):
        hist_species1 = hist[species1]
        hist_species2 = hist[species2]
        
        distances = []
        most_common1 = []
        most_common2 = []
        
        for location_code in range(len(hist_species1)):
            hist_species1_location = hist_species1[location_code]
            hist_species2_location = hist_species2[location_code]
            
            hist_species1_location_histogram = torch.histc(torch.Tensor(hist_species1_location), bins=self.possible_codes, min=0, max=self.possible_codes-1)
            hist_species1_location_histogram = hist_species1_location_histogram/torch.sum(hist_species1_location_histogram)
            most_common1.append(torch.argmax(hist_species1_location_histogram))
            
            hist_species2_location_histogram = torch.histc(torch.Tensor(hist_species2_location), bins=self.possible_codes, min=0, max=self.possible_codes-1)
            hist_species2_location_histogram = hist_species2_location_histogram/torch.sum(hist_species2_location_histogram)
            most_common2.append(torch.argmax(hist_species2_location_histogram))
        
        
            d = self.distance_used(hist_species1_location_histogram, hist_species2_location_histogram)
            distances.append(d)
    
        distances = torch.tensor(distances)
        most_common1 = torch.tensor(most_common1)
        most_common2 = torch.tensor(most_common2)
        
        return distances, most_common1, most_common2


@torch.no_grad()
def main(configs_yaml):
    yaml_path = configs_yaml.yaml_path
    ckpt_path = configs_yaml.ckpt_path
    DEVICE = configs_yaml.DEVICE
    distance_used = configs_yaml.distance_used
    file_list_path = configs_yaml.file_list_path    
    size = configs_yaml.size
    
    dataset = CustomDataset(size, file_list_path, add_labels=True)

    # Load model
    config = load_config(yaml_path, display=False)
    model = load_phylovqvae(config, ckpt_path=ckpt_path, data=dataset.data, cuda=(DEVICE is not None), model_type=VQModel)



    histograms_file = os.path.join(get_fig_pth(ckpt_path, postfix=CONSTANTS.HISTOGRAMS_FOLDER), CONSTANTS.HISTOGRAMS_FILE)
    histogram_file_exists = os.path.exists(histograms_file)
    if not histogram_file_exists:
        raise "histograms have not been generated. Run code_histogram.py first! Defaulting to index ordering"
    hist_arr, _ = pickle.load(open(histograms_file, "rb"))
    
    hist_parser = HistogramParser_VQGAN(model, DISTANCE_DICT[distance_used])
    
    jsdistances = torch.zeros([len(hist_arr), len(hist_arr)])
    for species1_indx in range(len(hist_arr)):
        for species2_indx in range(len(hist_arr)):
            if species2_indx<species1_indx:
                continue
            
            attr_distances, _, _ = hist_parser.get_distances(hist_arr, species1_indx, species2_indx)
            average_distance_attr = torch.mean(attr_distances)
            jsdistances[species1_indx, species2_indx]= jsdistances[species2_indx, species1_indx] = average_distance_attr
            
    plot_heatmap(jsdistances[:,:,].cpu(), ckpt_path, title='{} for phylo attributes'.format(distance_used), postfix=CONSTANTS.TEST_DIR)
            
            
            
            
    
    
    
    
  

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-n",
        "--config",
        type=str,
        nargs="?",
        const=True,
        default="analysis/configs/generate_histograms_matrix_vqgan.yaml",
    )
    
    cfg, _ = parser.parse_known_args()
    # cfg = parser.config
    configs = OmegaConf.load(cfg.config)
    cli = OmegaConf.from_cli()
    config = OmegaConf.merge(configs, cli)
    print(config)
    
    main(config)