from scripts.loading_utils import load_config, load_model
from scripts.analysis_utils import Embedding_Code_converter, HistogramParser
from scripts.plotting_utils import get_fig_pth, plot_heatmap
import scripts.constants as CONSTANTS

import torch
import os
from omegaconf import OmegaConf
import argparse
import pickle

#*****************

@torch.no_grad()
def main(configs_yaml):
    yaml_path = configs_yaml.yaml_path
    ckpt_path = configs_yaml.ckpt_path
    DEVICE = configs_yaml.DEVICE

    # Load model
    config = load_config(yaml_path, display=False)
    model = load_model(config, ckpt_path=ckpt_path, cuda=(DEVICE is not None))

    # load histograms
    histograms_file = os.path.join(get_fig_pth(ckpt_path, postfix=CONSTANTS.HISTOGRAMS_FOLDER), CONSTANTS.HISTOGRAMS_FILE)
    histogram_file_exists = os.path.exists(histograms_file)
    if not histogram_file_exists:
        raise "histograms have not been generated. Run code_histogram.py first! Defaulting to index ordering"
    hist_arr, hist_arr_nonattr = pickle.load(open(histograms_file, "rb"))
    
    # parse histograms and create phylo converter
    hist_parser = HistogramParser(model)
    converter = Embedding_Code_converter(model.phylo_disentangler.quantize.get_codebook_entry_index, model.phylo_disentangler.quantize.embedding, (1, model.phylo_disentangler.embed_dim, hist_parser.codes_per_phylolevel, hist_parser.n_phylolevels))
    
    # claculate distances
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
                sub_distances = converter.get_sub_level(attr_distances.unsqueeze(0), i)
                jsdistances[species1_indx, species2_indx, i+1]= jsdistances[species2_indx, species1_indx, i+1] = torch.mean(sub_distances)
                
            average_distance_attr = torch.mean(attr_distances)
            jsdistances[species1_indx, species2_indx, -1]= jsdistances[species2_indx, species1_indx, -1] = average_distance_attr
            
    plot_heatmap(jsdistances[:,:,0].cpu(), ckpt_path, title='js-divergence for non-attributes', postfix=CONSTANTS.TEST_DIR)
    for i in range(hist_parser.n_phylolevels-1):
        plot_heatmap(jsdistances[:,:,i+1].cpu(), ckpt_path, title='js-divergence for phylo attributes for level {}'.format(i), postfix=CONSTANTS.TEST_DIR)
    plot_heatmap(jsdistances[:,:,-1].cpu(), ckpt_path, title='js-divergence for phylo attributes', postfix=CONSTANTS.TEST_DIR)
            
            
            
            
    
    
    
    
  

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
    configs = OmegaConf.load(cfg.config)
    cli = OmegaConf.from_cli()
    config = OmegaConf.merge(configs, cli)
    print(config)
    
    main(config)