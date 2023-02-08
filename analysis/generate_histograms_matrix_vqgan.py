from scripts.analysis_utils import js_divergence
from scripts.loading_utils import load_config, load_model
from scripts.models.vqgan import VQModel
from scripts.plotting_utils import get_fig_pth, plot_heatmap
import scripts.constants as CONSTANTS

import torch
import os
from omegaconf import OmegaConf
import argparse
import pickle

#*****************

class HistogramParser_VQGAN:
    def __init__(self, model):
        self.possible_codes = model.quantize.n_e
    
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
        
        
            d = js_divergence(hist_species1_location_histogram, hist_species2_location_histogram)
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

    # Load model
    config = load_config(yaml_path, display=False)
    model = load_model(config, ckpt_path=ckpt_path, cuda=(DEVICE is not None), model_type=VQModel)

    # get histograms
    histograms_file = os.path.join(get_fig_pth(ckpt_path, postfix=CONSTANTS.HISTOGRAMS_FOLDER), CONSTANTS.HISTOGRAMS_FILE)
    histogram_file_exists = os.path.exists(histograms_file)
    if not histogram_file_exists:
        raise "histograms have not been generated. Run code_histogram.py first! Defaulting to index ordering"
    hist_arr, _ = pickle.load(open(histograms_file, "rb"))
    
    # parse histograms
    hist_parser = HistogramParser_VQGAN(model)
    
    # get js distance
    jsdistances = torch.zeros([len(hist_arr), len(hist_arr)])
    for species1_indx in range(len(hist_arr)):
        for species2_indx in range(len(hist_arr)):
            if species2_indx<species1_indx:
                continue
            
            attr_distances, _, _ = hist_parser.get_distances(hist_arr, species1_indx, species2_indx)
            average_distance_attr = torch.mean(attr_distances)
            jsdistances[species1_indx, species2_indx]= jsdistances[species2_indx, species1_indx] = average_distance_attr
            
    plot_heatmap(jsdistances[:,:,].cpu(), ckpt_path, title='js-divergence for phylo attributes', postfix=CONSTANTS.TEST_DIR)
            
            
            
            
    
    
    
    
  

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
    configs = OmegaConf.load(cfg.config)
    cli = OmegaConf.from_cli()
    config = OmegaConf.merge(configs, cli)
    print(config)
    
    main(config)