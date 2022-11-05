from taming.loading_utils import load_config, load_phylovqvae
from taming.analysis_utils import DISTANCE_DICT, Embedding_Code_converter, HistogramParser
from taming.plotting_utils import get_fig_pth, plot_heatmap

import torch

import taming.constants as CONSTANTS

import os

from omegaconf import OmegaConf
import argparse

import pickle

GENERATED_FOLDER = "most_likely_generations"

#*****************

@torch.no_grad()
def main(configs_yaml):
    yaml_path = configs_yaml.yaml_path
    ckpt_path = configs_yaml.ckpt_path
    DEVICE = configs_yaml.DEVICE
    distance_used = configs_yaml.distance_used

    # Load model
    config = load_config(yaml_path, display=False)
    model = load_phylovqvae(config, ckpt_path=ckpt_path).to(DEVICE)
    model.set_test_chkpt_path(ckpt_path)
    

    histograms_file = os.path.join(get_fig_pth(ckpt_path, postfix=CONSTANTS.HISTOGRAMS_FOLDER), CONSTANTS.HISTOGRAMS_FILE)
    histogram_file_exists = os.path.exists(histograms_file)
    if not histogram_file_exists:
        raise "histograms have not been generated. Run code_histogram.py first! Defaulting to index ordering"
    hist_arr, hist_arr_nonattr = pickle.load(open(histograms_file, "rb"))
    
    hist_parser = HistogramParser(model, DISTANCE_DICT[distance_used])
    converter = Embedding_Code_converter(model.phylo_disentangler.quantize.get_codebook_entry_index, model.phylo_disentangler.quantize.embedding, (1, model.phylo_disentangler.embed_dim, hist_parser.codebooks_per_phylolevel, hist_parser.n_phylolevels))
    
    jsdistances = torch.zeros([len(hist_arr), len(hist_arr), hist_parser.n_phylolevels+1])
    for species1_indx in range(len(hist_arr)):
        for species2_indx in range(len(hist_arr)):
            if species2_indx<species1_indx:
                continue
            
            attr_distances, _, _ = hist_parser.get_distances(hist_arr, species1_indx, species2_indx)
            nonattr_distances, _, _ = hist_parser.get_distances(hist_arr_nonattr, species1_indx, species2_indx)
            
            average_distance_nonattr = torch.mean(nonattr_distances)
            jsdistances[species1_indx, species2_indx, 0] = jsdistances[species2_indx, species1_indx, 0] = average_distance_nonattr
            
            for i in range(hist_parser.n_phylolevels-1):
                sub_distances = converter.reshape_code(converter.reshape_code(attr_distances.unsqueeze(0), reverse = True)[:,:,:i+1])
                jsdistances[species1_indx, species2_indx, i+1]= jsdistances[species2_indx, species1_indx, i+1] = torch.mean(sub_distances)
                
            average_distance_attr = torch.mean(attr_distances)
            jsdistances[species1_indx, species2_indx, -1]= jsdistances[species2_indx, species1_indx, -1] = average_distance_attr
            
    plot_heatmap(jsdistances[:,:,0].cpu(), ckpt_path, title='{} for non-attributes'.format(distance_used), postfix=CONSTANTS.TEST_DIR)
    for i in range(hist_parser.n_phylolevels-1):
        plot_heatmap(jsdistances[:,:,i+1].cpu(), ckpt_path, title='{} for phylo attributes for level {}'.format(distance_used, i), postfix=CONSTANTS.TEST_DIR)
    plot_heatmap(jsdistances[:,:,-1].cpu(), ckpt_path, title='{} for phylo attributes'.format(distance_used), postfix=CONSTANTS.TEST_DIR)
            
            
            
            
    
    
    
    
  

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-n",
        "--config",
        type=str,
        nargs="?",
        const=True,
        default="analysis/configs/generate_histograms_matrix.yaml",
    )
    
    cfg, _ = parser.parse_known_args()
    # cfg = parser.config
    configs = OmegaConf.load(cfg.config)
    cli = OmegaConf.from_cli()
    config = OmegaConf.merge(configs, cli)
    print(config)
    
    main(config)