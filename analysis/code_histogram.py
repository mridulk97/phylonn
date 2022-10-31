from taming.loading_utils import load_config, load_phylovqvae
from taming.data.custom import CustomTest as CustomDataset
from taming.data.utils import custom_collate
from taming.analysis_utils import Embedding_Code_converter
from taming.plotting_utils import get_fig_pth, save_image_grid, Histogram_plotter, save_to_cvs, save_to_txt

from torch.utils.data import DataLoader
import torch
from tqdm import tqdm

import taming.constants as CONSTANTS

import matplotlib.pyplot as plt

import os

from omegaconf import OmegaConf
import argparse

import random
import pickle
import csv
from operator import add

GENERATED_FOLDER = "most_likely_generations"

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
    num_specimen_generated = configs_yaml.num_specimen_generated
    create_histograms = configs_yaml.create_histograms
    per_phylo_level = configs_yaml.per_phylo_level
    generate_specimen = configs_yaml.generate_specifmen

    # Load model
    config = load_config(yaml_path, display=False)
    model = load_phylovqvae(config, ckpt_path=ckpt_path).to(DEVICE)
    model.set_test_chkpt_path(ckpt_path)

    # load image
    dataset = CustomDataset(size, file_list_path, add_labels=True)
    dataloader = DataLoader(dataset.data, batch_size=batch_size, num_workers=num_workers, collate_fn=custom_collate)

    converter = None
    hist_arr = [[] for x in range(len(model.phylo_disentangler.loss_phylo.phylogeny.getLabelList()))]
    
    passthrough = (model.phylo_disentangler.ch != model.phylo_disentangler.n_phylo_channels)
    anticlassification = (model.phylo_disentangler.loss_anticlassification )
    if anticlassification:
        hist_arr_nonattr = [[] for x in range(len(model.phylo_disentangler.loss_phylo.phylogeny.getLabelList()))]
    else:
        hist_arr_nonattr = {}
        
    # create the converter.
    item = next(iter(dataloader))
    img = model.get_input(item, model.image_key).to(DEVICE)
    lbl = item[CONSTANTS.DISENTANGLER_CLASS_OUTPUT]
    dec_image, _, _, in_out_disentangler = model(img)
    q_phylo_output = in_out_disentangler[CONSTANTS.QUANTIZED_PHYLO_OUTPUT]
    converter = Embedding_Code_converter(model.phylo_disentangler.quantize.get_codebook_entry_index, model.phylo_disentangler.quantize.embedding, q_phylo_output[0, :, :, :].shape)
    if anticlassification:
        q_phylo_output_nonattribute = in_out_disentangler[CONSTANTS.QUANTIZED_PHYLO_NONATTRIBUTE_OUTPUT]
        converter_nonattribute = Embedding_Code_converter(model.phylo_disentangler.quantize.get_codebook_entry_index, model.phylo_disentangler.quantize.embedding, q_phylo_output_nonattribute[0, :, :, :].shape)
            
                    
        
    # collect values   
    hist_file_path = os.path.join(get_fig_pth(ckpt_path, postfix=CONSTANTS.HISTOGRAMS_FOLDER), CONSTANTS.HISTOGRAMS_FILE)
    try:
        hist_arr, hist_arr_nonattr = pickle.load(open(hist_file_path, "rb"))
        print(hist_file_path, 'loaded!')
    except (OSError, IOError) as e:
        print(hist_file_path, '... Calculating histograms.')
        for item in tqdm(dataloader):
            img = model.get_input(item, model.image_key).to(DEVICE)
            lbl = item[CONSTANTS.DISENTANGLER_CLASS_OUTPUT]
            
            # get output
            dec_image, _, _, in_out_disentangler = model(img)
            q_phylo_output = in_out_disentangler[CONSTANTS.QUANTIZED_PHYLO_OUTPUT]
            if anticlassification :
                q_phylo_output_nonattribute = in_out_disentangler[CONSTANTS.QUANTIZED_PHYLO_NONATTRIBUTE_OUTPUT]
            
            # reshape
            q_phylo_output_indices = converter.get_phylo_codes(q_phylo_output)
            if anticlassification:
                q_phylo_nonattribute_output_indices = converter_nonattribute.get_phylo_codes(q_phylo_output_nonattribute)
            
            if len(hist_arr[0]) == 0:
                for i in range(len(hist_arr)):
                    hist_arr[i] = [[] for x in range(q_phylo_output_indices.shape[1])]
                    if anticlassification :
                        hist_arr_nonattr[i] = [[] for x in range(q_phylo_nonattribute_output_indices.shape[1])]
            
            # iterate through code locations
            for code_location in range(q_phylo_output_indices.shape[1]):
                code = q_phylo_output_indices[0, code_location]
                hist_arr[lbl][code_location].append(code.item()) 
            if anticlassification:
                for code_location in range(q_phylo_nonattribute_output_indices.shape[1]):
                    code = q_phylo_nonattribute_output_indices[0, code_location]
                    hist_arr_nonattr[lbl][code_location].append(code.item())
        
        pickle.dump((hist_arr, hist_arr_nonattr), open(hist_file_path, "wb"))
        print(hist_file_path, 'saved!')
         
         
    
    codebooks_per_phylolevel = model.phylo_disentangler.codebooks_per_phylolevel
    n_phylolevels = model.phylo_disentangler.n_phylolevels
    if anticlassification:
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
        
        group_levels_attr_hist, group_levels_non_attr_hist = build_group_histograms(hist_arr, hist_arr_nonattr,
                            species_groups_arr, n_phylolevels,
                            dataset.labels_to_idx,
                            anticlassification=anticlassification)
    
        
            

    # create histograms
    if create_histograms:
        print('plotting histograms...')
        
        hist_plotter = Histogram_plotter(codebooks_per_phylolevel, n_phylolevels, model.phylo_disentangler.n_embed, converter, dataset.indx_to_label, ckpt_path, CONSTANTS.HISTOGRAMS_FOLDER)
        if anticlassification:
            hist_plotter_non_attribute = Histogram_plotter(codebooks_per_phylolevel, n_levels_nonattribute, model.phylo_disentangler.n_embed, converter_nonattribute, dataset.indx_to_label, ckpt_path, CONSTANTS.HISTOGRAMS_FOLDER)
        
        for species_indx, species_arr in tqdm(enumerate(hist_arr)):         
            hist_plotter.plot_histograms(species_arr, species_indx, is_nonattribute=False)
            
            if anticlassification:
                hist_plotter_non_attribute.plot_histograms(hist_arr_nonattr[species_indx], species_indx, is_nonattribute=True)
        
        
        
        if per_phylo_level:
            for level in range(n_phylolevels-1): # last level already plotted.

                for species_group in tqdm(species_groups_arr[level]):
                    
                    hist_plotter.plot_histograms(group_levels_attr_hist[level][species_group[0]], dataset.labels_to_idx[species_group[0]], is_nonattribute=False, prefix="group-level-{}".format(level))
                    
                    if anticlassification :
                        hist_plotter_non_attribute.plot_histograms(group_levels_non_attr_hist[level][species_group[0]], dataset.labels_to_idx[species_group[0]], is_nonattribute=True, prefix="group-level-{}".format(level))
            
                    
                        
                
    # create likely species if we are in the right model.  
    if generate_specimen: 
        print('generating images...')
        
        if ((not passthrough) or anticlassification):
            
            # For all species.
            for species_indx, species_arr in tqdm(enumerate(hist_arr)):  
            
                generate_images(species_arr, species_indx,
                        hist_arr_nonattr[species_indx], 
                        num_specimen_generated, 
                        model, converter, converter_nonattribute, dataset.indx_to_label,
                        DEVICE, ckpt_path, 'species',
                        anticlassification=anticlassification)
                
            if per_phylo_level:
                for level in range(n_phylolevels-1): # last level already plotted.
                    for species_group in tqdm(species_groups_arr[level]):
                        
                        generate_images(group_levels_attr_hist[level][species_group[0]], dataset.labels_to_idx[species_group[0]],
                            group_levels_non_attr_hist[level][species_group[0]], 
                            num_specimen_generated, 
                            model, converter, converter_nonattribute, dataset.indx_to_label,
                            DEVICE, ckpt_path, 'group-level-{}'.format(level),
                            anticlassification=anticlassification)
        else:
            print("We won't create likely images because of species because there is a passthrough without anticlassification")



# Gives structure of shape:
# [num_of_levels-1][species][code]
def build_group_histograms(hist_arr, hist_arr_nonattr,
                           species_groups_arr, num_of_levels,
                           labels_to_idx,
                           anticlassification=False):
    group_levels_attr = []
    group_levels_non_attr = []
    
    for level in range(num_of_levels-1): # last level already plotted.

        group_arr ={}
        if anticlassification:
            group_arr_nonattr ={}
        
        for species_group in species_groups_arr[level]:
            
            for indx, species in enumerate(species_group):
                species_indx = labels_to_idx[species]
                if indx == 0:
                    group_arr[species] = hist_arr[species_indx]
                    if anticlassification:
                        group_arr_nonattr[species] = hist_arr_nonattr[species_indx]
                else:
                    group_arr[species_group[0]] = list(map(add, group_arr[species_group[0]], hist_arr[species_indx]))
                    if anticlassification:
                        group_arr_nonattr[species_group[0]] =  list(map(add, group_arr_nonattr[species_group[0]], hist_arr_nonattr[species_indx]))
                        
        group_levels_attr.append(group_arr)
        if anticlassification:
            group_levels_non_attr.append(group_arr_nonattr)
        
    return group_levels_attr, group_levels_non_attr
                        



def generate_images(species_arr, species_indx,
                    species_arr_nonattr, 
                    num_specimen_generated, 
                    model, converter, converter_nonattribute, indx_to_label,
                    device, ckpt_path, prefix_text,
                    anticlassification=False):
    
    
    list_of_created_sequence = []
    list_of_created_nonattribute_sequence = []     
            
    # for all images
    generated_imgs = []
    for i in range(num_specimen_generated):
        created_sequence = torch.zeros((1, len(species_arr))).to(device).long()
        if anticlassification:
            created_nonattribute_sequence = torch.zeros((1, len(species_arr_nonattr))).to(device).long()
        
        # for all code locations
        for j in range(len(species_arr)):
            
            # generate the sequences.
            code = random.choice(species_arr[j])
            created_sequence[0, j] = code
        
        if anticlassification:
            for j in range(len(species_arr_nonattr)):
                code_nonattribute = random.choice(species_arr_nonattr[j])
                created_nonattribute_sequence[0, j] = code_nonattribute
                
        # append the generated sequence
        list_of_created_sequence.append(created_sequence.reshape(-1).tolist())
        if anticlassification:
            list_of_created_nonattribute_sequence.append(created_nonattribute_sequence.reshape(-1).tolist())
            
        embedding = converter.get_phylo_embeddings(created_sequence)
        embedding_nonattribute = converter_nonattribute.get_phylo_embeddings(created_nonattribute_sequence) if anticlassification else None
        dec_image_new, _ = model.from_quant_only(embedding, embedding_nonattribute)
        generated_imgs.append(dec_image_new)
        
    # Save the sequences
    save_to_cvs(ckpt_path, GENERATED_FOLDER, "{}_attributecodes_{}_{}.csv".format(prefix_text, species_indx, indx_to_label[species_indx]), list_of_created_sequence)
    if anticlassification:
        save_to_cvs(ckpt_path, GENERATED_FOLDER, "{}_non_attributecodes_{}_{}.csv".format(prefix_text, species_indx, indx_to_label[species_indx]), list_of_created_nonattribute_sequence)

    # save the images
    generated_imgs = torch.cat(generated_imgs, dim=0)
    save_image_grid(generated_imgs, ckpt_path, subfolder=GENERATED_FOLDER, postfix= "{}_{}_{}".format(prefix_text, species_indx, indx_to_label[species_indx]))



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
    # cfg = parser.config
    configs = OmegaConf.load(cfg.config)
    cli = OmegaConf.from_cli()
    config = OmegaConf.merge(configs, cli)
    print(config)
    
    main(config)