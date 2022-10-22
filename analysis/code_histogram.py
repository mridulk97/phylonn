from taming.loading_utils import load_config, load_phylovqvae
from taming.data.custom import CustomTest as CustomDataset
from taming.data.utils import custom_collate
from taming.analysis_utils import Embedding_Code_converter
from taming.plotting_utils import get_fig_pth, save_image_grid

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

HISTOGRAMS_FILE="histograms.pkl"
HISTOGRAMS_FOLDER='code_histograms'
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
    generate_specifmen = configs_yaml.generate_specifmen

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
    anticlassification = model.phylo_disentangler.loss_anticlassification
    if passthrough and (anticlassification is not None):
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
                    
        
    # collect values   
    hist_file_path = os.path.join(get_fig_pth(ckpt_path, postfix=HISTOGRAMS_FOLDER), HISTOGRAMS_FILE)
    try:
        hist_arr, hist_arr_nonattr = pickle.load(open(hist_file_path, "rb"))
        print(hist_file_path, 'loaded!')
    except (OSError, IOError) as e:
        print(hist_file_path, '... Calculating.')
        for item in tqdm(dataloader):
            img = model.get_input(item, model.image_key).to(DEVICE)
            lbl = item[CONSTANTS.DISENTANGLER_CLASS_OUTPUT]
            
            # get output
            dec_image, _, _, in_out_disentangler = model(img)
            q_phylo_output = in_out_disentangler[CONSTANTS.QUANTIZED_PHYLO_OUTPUT]
            if passthrough and (anticlassification is not None):
                q_phylo_output_nonattribute = in_out_disentangler[CONSTANTS.QUANTIZED_PHYLO_NONATTRIBUTE_OUTPUT]
            
            # reshape
            if converter is None:
                converter = Embedding_Code_converter(model.phylo_disentangler.quantize.get_codebook_entry_index, model.phylo_disentangler.quantize.embedding, q_phylo_output[0, :, :, :].shape)
            q_phylo_output_indices = converter.get_phylo_codes(q_phylo_output)
            if passthrough and (anticlassification is not None):
                q_phylo_nonattribute_output_indices = converter.get_phylo_codes(q_phylo_output_nonattribute)
            
            if len(hist_arr[0]) == 0:
                for i in range(len(hist_arr)):
                    hist_arr[i] = [[] for x in range(q_phylo_output_indices.shape[1])]
                    if passthrough and (anticlassification is not None):
                        hist_arr_nonattr[i] = [[] for x in range(q_phylo_nonattribute_output_indices.shape[1])]
            
            # iterate through code locations
            for code_location in range(q_phylo_output_indices.shape[1]):
                code = q_phylo_output_indices[0, code_location]
                hist_arr[lbl][code_location].append(code.item()) 
            if passthrough and (anticlassification is not None):
                for code_location in range(q_phylo_nonattribute_output_indices.shape[1]):
                    code = q_phylo_nonattribute_output_indices[0, code_location]
                    hist_arr_nonattr[lbl][code_location].append(code.item())
        
        pickle.dump((hist_arr, hist_arr_nonattr), open(hist_file_path, "wb"))
        print(hist_file_path, 'saved!')
         
         
    
    codebooks_per_phylolevel = model.phylo_disentangler.codebooks_per_phylolevel
    n_phylolevels = model.phylo_disentangler.n_phylolevels

    # create histograms
    if create_histograms:
        for species_indx, species_arr in tqdm(enumerate(hist_arr)):         
            
            fig, axs = plt.subplots(codebooks_per_phylolevel, n_phylolevels, figsize = (20,30))
            for i, ax in enumerate(axs.reshape(-1)):
                ax.hist(species_arr[i], density=True, range=(0, model.phylo_disentangler.n_embed-1), bins=model.phylo_disentangler.n_embed)
                code_location, level = converter.get_code_reshaped_index(i)
                ax.set_title("code "+ str(code_location) + "/level " +str(level))
            plt.show()
            fig.savefig(os.path.join(get_fig_pth(ckpt_path, postfix=HISTOGRAMS_FOLDER+'/attribute'), "species_{}_{}_hostogram.png".format(species_indx, dataset.indx_to_label[species_indx])))
            plt.close(fig)
            
            if passthrough and (anticlassification is not None):
                fig, axs = plt.subplots(codebooks_per_phylolevel, n_phylolevels, figsize = (20,30))
                for i, ax in enumerate(axs.reshape(-1)):
                    ax.hist(hist_arr_nonattr[species_indx][i], density=True, range=(0, model.phylo_disentangler.n_embed-1), bins=model.phylo_disentangler.n_embed)
                    ax.set_title("code "+ str(i))
                plt.show()
                fig.savefig(os.path.join(get_fig_pth(ckpt_path, postfix=HISTOGRAMS_FOLDER+'/non_attribute'), "species_{}_{}_hostogram.png".format(species_indx, dataset.indx_to_label[species_indx])))
                plt.close(fig)
            
        
    
    # create likely species if we are in the right model.  
    if generate_specifmen: 
        if ((not passthrough) or (anticlassification is not None)):
            for species_indx, species_arr in tqdm(enumerate(hist_arr)):  
                list_of_created_sequence = []
                list_of_created_nonattribute_sequence = []     
                        
                # for all images
                generated_imgs = []
                for i in range(num_specimen_generated):
                    # for all code locations
                    created_sequence = torch.zeros((1, len(species_arr))).to(DEVICE).long()
                    if anticlassification:
                        created_nonattribute_sequence = torch.zeros((1, len(hist_arr_nonattr[species_indx]))).to(DEVICE).long()
                    for j in range(len(species_arr)):
                        code = random.choice(species_arr[j])
                        created_sequence[0, j] = code
                        if anticlassification:
                            code_nonattribute = random.choice(hist_arr_nonattr[species_indx][j])
                            created_nonattribute_sequence[0, j] = code_nonattribute
                            
                    list_of_created_sequence.append(created_sequence.reshape(-1).tolist())
                    if anticlassification:
                        list_of_created_nonattribute_sequence.append(created_nonattribute_sequence.reshape(-1).tolist())
                        
                    embedding = converter.get_phylo_embeddings(created_sequence)
                    embedding_nonattribute = converter.get_phylo_embeddings(created_nonattribute_sequence) if anticlassification else None
                    dec_image_new, _ = model.from_quant_only(embedding, embedding_nonattribute)
                    generated_imgs.append(dec_image_new)
                    
                file = open(os.path.join(get_fig_pth(ckpt_path, postfix=GENERATED_FOLDER), "species_attributecodes_{}_{}.csv".format(species_indx, dataset.indx_to_label[species_indx])), 'w')
                with file:  
                    write = csv.writer(file)
                    write.writerows(list_of_created_sequence)
                file = open(os.path.join(get_fig_pth(ckpt_path, postfix=GENERATED_FOLDER), "species_non_attributecodes_{}_{}.csv".format(species_indx, dataset.indx_to_label[species_indx])), 'w')
                if anticlassification:
                    with file:  
                        write = csv.writer(file)
                        write.writerows(list_of_created_nonattribute_sequence)
                
                generated_imgs = torch.cat(generated_imgs, dim=0)
                save_image_grid(generated_imgs, ckpt_path, subfolder=GENERATED_FOLDER, postfix= "species_{}_{}".format(species_indx, dataset.indx_to_label[species_indx]))

        else:
            print("We won't create likely images because of species because there is a passthrough without anticlassification")







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