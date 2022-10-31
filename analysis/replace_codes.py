

from taming.loading_utils import load_config, load_phylovqvae
from taming.plotting_utils import save_image_grid, get_fig_pth
from taming.data.custom import CustomTest as CustomDataset
from taming.data.utils import custom_collate
from taming.analysis_utils import Embedding_Code_converter

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from scipy.stats import entropy

import taming.constants as CONSTANTS

from omegaconf import OmegaConf
import argparse

import os
import pickle

####################

class Clone_manger():
    def __init__(self, cummulative, embedding):#, num_of_codes):
        self.cummulative = cummulative
        self.embedding = embedding

    def get_embedding(self, index=None):
        assert (index is not None) or (not self.cummulative)
        if not self.cummulative:
            return torch.clone(self.embedding)
        else:
            # return self.clone_list[index]
            return self.embedding


# indexing of hist_arr: [code_location][raw list of codes of that location from all images]
# highest entropy to lowest entropy
def get_entropy_ordering(hist_arr_for_species):
    entropies = []
    for codes_forcode_location in hist_arr_for_species:
        value,counts = np.unique(codes_forcode_location, return_counts=True)
        entropies.append(entropy(counts))
    reverse_ordered_entropy_indices = np.argsort(entropies)[::-1]
    # print(entropies, reverse_ordered_entropy_indices)
    return reverse_ordered_entropy_indices
    
# From least frequent to most frequent
def get_highest_likelyhood_ordering(hist_arr_for_species_and_location):
    value,counts = np.unique(hist_arr_for_species_and_location, return_counts=True)
    ordered_count_indices = np.argsort(counts)
    # print(counts, ordered_count_indices)
    return ordered_count_indices
    


@torch.no_grad()
def main(configs_yaml):
    yaml_path = configs_yaml.yaml_path
    ckpt_path = configs_yaml.ckpt_path
    DEVICE = configs_yaml.DEVICE
    image_index = configs_yaml.image_index
    file_list_path = configs_yaml.file_list_path
    size = configs_yaml.size
    cummulative = configs_yaml.cummulative
    plot_diff = configs_yaml.plot_diff
    by_entropy = configs_yaml.by_entropy

    # Load model
    config = load_config(yaml_path, display=False)
    model = load_phylovqvae(config, ckpt_path=ckpt_path).to(DEVICE)
    model.set_test_chkpt_path(ckpt_path)

    # load image
    dataset = CustomDataset(size, file_list_path, add_labels=True)
    specimen = dataset[image_index]
    processed_img = specimen['image']
    processed_img = torch.from_numpy(processed_img).unsqueeze(0)
    processed_img = processed_img.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format)
    processed_img = processed_img.float()
    species_index = specimen['class']
    print(specimen['file_path_'], specimen['class'])

    # get output
    dec_image, _, _, in_out_disentangler = model(processed_img.to(DEVICE))
    q_phylo_output = in_out_disentangler[CONSTANTS.QUANTIZED_PHYLO_OUTPUT]

    converter = Embedding_Code_converter(model.phylo_disentangler.quantize.get_codebook_entry_index, model.phylo_disentangler.quantize.embedding, q_phylo_output[0, :, :, :].shape)
    all_code_indices = converter.get_phylo_codes(q_phylo_output[0, :, :, :].unsqueeze(0), verify=False)
    all_codes_reverse_reshaped = converter.get_phylo_embeddings(all_code_indices, verify=False)

    dec_image_reversed, _, _, _ = model(processed_img.to(DEVICE), overriding_quant=all_codes_reverse_reshaped)
    
    # Get code and location ordering
    histograms_file = os.path.join(get_fig_pth(ckpt_path, postfix=CONSTANTS.HISTOGRAMS_FOLDER), CONSTANTS.HISTOGRAMS_FILE)
    histogram_file_exists = os.path.exists(histograms_file)
    if by_entropy and not histogram_file_exists:
        print("histograms have not been generated. Run code_histogram.py first! Defaulting to index ordering")
    using_entropy = by_entropy and os.path.exists(histograms_file)
    if not using_entropy:
        which_codes = range(model.phylo_disentangler.n_embed)
        which_locations = range(model.phylo_disentangler.codebooks_per_phylolevel*model.phylo_disentangler.n_phylolevels)
    else:
        hist_arr, hist_arr_nonattr = pickle.load(open(histograms_file, "rb"))
        which_locations = get_entropy_ordering(hist_arr[species_index])
        
        
    
    clone_manager = Clone_manger(cummulative, all_codes_reverse_reshaped) #  model.phylo_disentangler.n_embed


    # for level in range(model.phylo_disentangler.n_phylolevels):
    for ordering, code_level_location in enumerate(which_locations):
        code_location, level = converter.get_code_reshaped_index(code_level_location)
        
        generated_imgs = [dec_image, dec_image_reversed]
            
        if using_entropy:
            which_codes = get_highest_likelyhood_ordering(hist_arr[species_index][code_level_location])
            for i in range(model.phylo_disentangler.n_embed):
                if i not in which_codes:
                    which_codes = np.insert(which_codes, 0, i)#np.append(which_codes, i)
            
        for code_index in which_codes:
            all_codes_reverse_reshaped_clone = clone_manager.get_embedding(code_index)
            all_codes_reverse_reshaped_clone[0, :, code_location, level] = model.phylo_disentangler.quantize.embedding(torch.tensor([code_index]).to(all_codes_reverse_reshaped.device))

            dec_image_new, _, _, _ = model(processed_img.to(DEVICE), overriding_quant=all_codes_reverse_reshaped_clone)
            
            if plot_diff:
                generated_imgs.append(dec_image_new - dec_image)
            else:
                generated_imgs.append(dec_image_new)
        
        generated_imgs = torch.cat(generated_imgs, dim=0)
        save_image_grid(generated_imgs, ckpt_path, subfolder="codebook_grid-cumulative{}-diff{}".format(cummulative, plot_diff), postfix="ordering{}-level{}-location{}".format(ordering, level, code_location))

        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-n",
        "--config",
        type=str,
        nargs="?",
        const=True,
        default="analysis/configs/replace_codes.yaml",
    )
    
    cfg, _ = parser.parse_known_args()
    # cfg = parser.config
    configs = OmegaConf.load(cfg.config)
    cli = OmegaConf.from_cli()
    config = OmegaConf.merge(configs, cli)
    print(config)
    
    main(config)