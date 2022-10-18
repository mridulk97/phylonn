from taming.loading_utils import load_config, load_phylovqvae
from taming.data.custom import CustomTest as CustomDataset
from taming.data.utils import custom_collate
from taming.analysis_utils import Embedding_Code_converter
from taming.plotting_utils import get_fig_pth

from torch.utils.data import DataLoader
import torch
from tqdm import tqdm

import taming.constants as CONSTANTS

import matplotlib.pyplot as plt

import os


##########################

DEVICE=0

num_workers = 8
batch_size = 1 # DO NOT CHANGE! 

size= 256
file_list_path = "/home/elhamod/data/Fish/taming_transforms_fish_train_padded_256.txt"

# size= 512
# file_list_path = "/home/elhamod/data/Fish/taming_transforms_fish_train_padded_512.txt"

# file_list_path = "/home/elhamod/data/Fish/taming_transforms_fish_test_small.txt" # Just for test


# # 256 combined nopass
# yaml_path = "/home/elhamod/projects/taming-transformers/logs/2022-10-14T10-40-42_Phylo-VQVAE256img-afterhyperp-combined/configs/2022-10-14T10-40-42-project.yaml"
# ckpt_path = "/home/elhamod/projects/taming-transformers/logs/2022-10-14T10-40-42_Phylo-VQVAE256img-afterhyperp-combined/checkpoints/last.ckpt"

# 4cbs
yaml_path = "/home/elhamod/projects/taming-transformers/logs/2022-10-17T12-20-24_Phylo-VQVAE256img-afterhyperp-combined-4cbperlevel/configs/2022-10-17T12-20-24-project.yaml"
ckpt_path = "/home/elhamod/projects/taming-transformers/logs/2022-10-17T12-20-24_Phylo-VQVAE256img-afterhyperp-combined-4cbperlevel/checkpoints/last.ckpt"


# # 46ch
# yaml_path = "/home/elhamod/projects/taming-transformers/logs/2022-10-14T00-55-06_Phylo-VQVAE256img-afterhyperp-combined/configs/2022-10-14T00-55-06-project.yaml"
# ckpt_path = "/home/elhamod/projects/taming-transformers/logs/2022-10-14T00-55-06_Phylo-VQVAE256img-afterhyperp-combined/checkpoints/last.ckpt"


##########

@torch.no_grad()
def main():

    # Load model
    config = load_config(yaml_path, display=False)
    model = load_phylovqvae(config, ckpt_path=ckpt_path).to(DEVICE)
    model.set_test_chkpt_path(ckpt_path)

    # load image
    dataset = CustomDataset(size, file_list_path, add_labels=True)
    dataloader = DataLoader(dataset.data, batch_size=batch_size, num_workers=num_workers, collate_fn=custom_collate)

    converter = None
    hist_arr = [[] for x in range(len(model.phylo_disentangler.loss_phylo.phylogeny.getLabelList()))]
        
    # collect values    
    for item in tqdm(dataloader):
        img = model.get_input(item, model.image_key).to(DEVICE)
        lbl = item[CONSTANTS.DISENTANGLER_CLASS_OUTPUT]
        
        # get output
        dec_image, _, _, in_out_disentangler = model(img)
        q_phylo_output = in_out_disentangler[CONSTANTS.QUANTIZED_PHYLO_OUTPUT]
        # reshape
        if converter is None:
            converter = Embedding_Code_converter(model.phylo_disentangler.quantize.get_codebook_entry_index, model.phylo_disentangler.quantize.embedding, q_phylo_output[0, :, :, :].shape)
        q_phylo_output_indices = converter.get_phylo_codes(q_phylo_output)
        
        if len(hist_arr[0]) == 0:
            for i in range(len(hist_arr)):
                 hist_arr[i] = [[] for x in range(q_phylo_output_indices.shape[1])]
        
        # iterate through code locations
        for code_location in range(q_phylo_output_indices.shape[1]):
            code = q_phylo_output_indices[0, code_location]
            
            hist_arr[lbl][code_location].append(code.item())
         
         
    
    codebooks_per_phylolevel = model.phylo_disentangler.codebooks_per_phylolevel
    n_phylolevels = model.phylo_disentangler.n_phylolevels

    for species_indx, species_arr in enumerate(hist_arr):         
        
        fig, axs = plt.subplots(codebooks_per_phylolevel, n_phylolevels, figsize = (20,20))
        
        for i, ax in enumerate(axs.reshape(-1)):
            ax.hist(species_arr[i], density=True, range=(0, model.phylo_disentangler.n_embed-1), bins=model.phylo_disentangler.n_embed)
        
        fig.savefig(os.path.join(get_fig_pth(ckpt_path, postfix='code_histograms'), "species_{}_{}_hostogram.png".format(species_indx, dataset.indx_to_label[species_indx])))
        
        plt.close(fig)

if __name__ == "__main__":
    main()