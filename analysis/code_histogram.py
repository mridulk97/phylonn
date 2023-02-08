from scripts.loading_utils import load_config, load_phylovqvae
from scripts.data.custom import CustomTest as CustomDataset
from scripts.data.utils import custom_collate
from scripts.analysis_utils import Embedding_Code_converter, HistogramFrequency
from scripts.plotting_utils import get_fig_pth, Histogram_plotter, save_to_txt
import scripts.constants as CONSTANTS

from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
import os
from omegaconf import OmegaConf
import argparse
from operator import add

##########

@torch.no_grad()
def main(configs_yaml):
    yaml_path = configs_yaml.yaml_path
    ckpt_path = configs_yaml.ckpt_path
    DEVICE = configs_yaml.DEVICE
    batch_size = configs_yaml.batch_size
    file_list_path = configs_yaml.file_list_path    
    num_workers = configs_yaml.num_workers
    size = configs_yaml.size
    create_histograms = configs_yaml.create_histograms
    per_phylo_level = configs_yaml.per_phylo_level

    unique_skipped_labels = configs_yaml.unique_skipped_labels

    # load image
    dataset = CustomDataset(size, file_list_path, add_labels=True, unique_skipped_labels=unique_skipped_labels)
    dataset_noskippedlabels = CustomDataset(size, file_list_path, add_labels=True)
    dataloader_noskippedlabels = DataLoader(dataset_noskippedlabels.data, batch_size=batch_size, num_workers=num_workers, collate_fn=custom_collate)
    
    # Load model
    config = load_config(yaml_path, display=False)
    model = load_phylovqvae(config, ckpt_path=ckpt_path, cuda=(DEVICE is not None))
        
    # create the converter.
    item = next(iter(dataloader_noskippedlabels))
    img = model.get_input(item, model.image_key).to(DEVICE)
    lbl = item[CONSTANTS.DISENTANGLER_CLASS_OUTPUT]
    _, _, _, in_out_disentangler = model(img)
    
    q_phylo_output = in_out_disentangler[CONSTANTS.QUANTIZED_PHYLO_OUTPUT]
    converter = Embedding_Code_converter(model.phylo_disentangler.quantize.get_codebook_entry_index, model.phylo_disentangler.quantize.embedding, q_phylo_output[0, :, :, :].shape)
    q_phylo_output_indices = converter.get_phylo_codes(q_phylo_output)

    q_phylo_output_nonattribute = in_out_disentangler[CONSTANTS.QUANTIZED_PHYLO_NONATTRIBUTE_OUTPUT]
    converter_nonattribute = Embedding_Code_converter(model.phylo_disentangler.quantize.get_codebook_entry_index, model.phylo_disentangler.quantize.embedding, q_phylo_output_nonattribute[0, :, :, :].shape)
    q_phylo_output_nonattribute_indices = converter.get_phylo_codes(q_phylo_output_nonattribute)
    
    # create histogram counter
    hist_freq = HistogramFrequency(len(dataset.indx_to_label.keys()), q_phylo_output_indices.shape[1], q_phylo_output_nonattribute_indices.shape[1])
                    
        
    # collect values   
    hist_file_path = os.path.join(get_fig_pth(ckpt_path, postfix=CONSTANTS.HISTOGRAMS_FOLDER), CONSTANTS.HISTOGRAMS_FILE)
    try:
        hist_freq.load_from_file(hist_file_path)
    except (OSError, IOError) as e:
        print(hist_file_path, '... Calculating histograms.')
        for item in tqdm(dataloader_noskippedlabels):
            img = model.get_input(item, model.image_key).to(DEVICE)
            lbl = item[CONSTANTS.DISENTANGLER_CLASS_OUTPUT]
            
            # get output
            _, _, _, in_out_disentangler = model(img)
            
            q_phylo_output = in_out_disentangler[CONSTANTS.QUANTIZED_PHYLO_OUTPUT]
            q_phylo_output_indices = converter.get_phylo_codes(q_phylo_output)
            
            q_phylo_output_nonattribute = in_out_disentangler[CONSTANTS.QUANTIZED_PHYLO_NONATTRIBUTE_OUTPUT]
            q_phylo_nonattribute_output_indices = converter_nonattribute.get_phylo_codes(q_phylo_output_nonattribute)
            
            hist_freq.set_location_frequencies(lbl, q_phylo_output_indices, q_phylo_nonattribute_output_indices)
        
        hist_freq.save_to_file(hist_file_path)
         
         
    
    codebooks_per_phylolevel = model.phylo_disentangler.codebooks_per_phylolevel
    n_phylolevels = model.phylo_disentangler.n_phylolevels
    n_levels_nonattribute = model.phylo_disentangler.n_levels_non_attribute if model.phylo_disentangler.n_levels_non_attribute is not None else model.phylo_disentangler.n_phylolevels
    
    if per_phylo_level: 
        print('generating group histograms...')
        species_groups_arr = []
        
        for level in range(n_phylolevels-1): # last level already plotted.
            relative_distance =  model.phylo_disentangler.loss_phylo.get_relative_distance_for_level(level)
            species_groups = model.phylo_disentangler.loss_phylo.phylogeny.get_species_groups(relative_distance)
            species_groups_arr.append(species_groups)
            
            species_groups_list = list(species_groups)
            save_to_txt(species_groups_list, ckpt_path, "species-groups-level-{}".format(level))
        
        group_levels_attr_hist, group_levels_non_attr_hist = build_group_histograms(hist_freq,
                            species_groups_arr, n_phylolevels,
                            dataset.labels_to_idx)
    
        
            

    # create histograms
    if create_histograms:
        print('plotting histograms...')
        
        hist_plotter = Histogram_plotter(codebooks_per_phylolevel, n_phylolevels, model.phylo_disentangler.n_embed, converter, dataset.indx_to_label, ckpt_path, CONSTANTS.HISTOGRAMS_FOLDER)
        hist_plotter_non_attribute = Histogram_plotter(codebooks_per_phylolevel, n_levels_nonattribute, model.phylo_disentangler.n_embed, converter_nonattribute, dataset.indx_to_label, ckpt_path, CONSTANTS.HISTOGRAMS_FOLDER)
        
        for species_indx, species_arr in tqdm(enumerate(hist_freq.hist_arr)):         
            hist_plotter.plot_histograms(species_arr, species_indx, is_nonattribute=False)
            hist_plotter_non_attribute.plot_histograms(hist_freq.hist_arr_nonattr[species_indx], species_indx, is_nonattribute=True)
        

        if per_phylo_level:
            for level in range(n_phylolevels-1): # last level already plotted.

                for species_group in tqdm(species_groups_arr[level]):
                    
                    hist_plotter.plot_histograms(group_levels_attr_hist[level][species_group[0]], dataset.labels_to_idx[species_group[0]], is_nonattribute=False, prefix="group-level-{}".format(level))
                    hist_plotter_non_attribute.plot_histograms(group_levels_non_attr_hist[level][species_group[0]], dataset.labels_to_idx[species_group[0]], is_nonattribute=True, prefix="group-level-{}".format(level))



# Gives structure of shape:
# [num_of_levels-1][species][code]
def build_group_histograms(hist_freq,
                           species_groups_arr, num_of_levels,
                           labels_to_idx):
    group_levels_attr = []
    group_levels_non_attr = []
    
    for level in range(num_of_levels-1): # last level already plotted.

        group_arr ={}
        group_arr_nonattr ={}
        
        for species_group in species_groups_arr[level]:
            
            for indx, species in enumerate(species_group):
                species_indx = labels_to_idx[species]
                if indx == 0:
                    group_arr[species] = hist_freq.hist_arr[species_indx]
                    group_arr_nonattr[species] = hist_freq.hist_arr_nonattr[species_indx]
                else:
                    group_arr[species_group[0]] = list(map(add, group_arr[species_group[0]], hist_freq.hist_arr[species_indx]))
                    group_arr_nonattr[species_group[0]] =  list(map(add, group_arr_nonattr[species_group[0]], hist_freq.hist_arr_nonattr[species_indx]))
                        
        group_levels_attr.append(group_arr)
        group_levels_non_attr.append(group_arr_nonattr)
        
    return group_levels_attr, group_levels_non_attr
                        


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-n",
        "--config",
        type=str,
        nargs="?",
        const=True,
        default="analysis/configs/code_histogram.yaml",
    )
    
    cfg, _ = parser.parse_known_args()
    configs = OmegaConf.load(cfg.config)
    cli = OmegaConf.from_cli()
    config = OmegaConf.merge(configs, cli)
    print(config)
    
    main(config)