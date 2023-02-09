import copy
from taming.loading_utils import load_config, load_phylovqvae
from taming.data.custom import CustomTest as CustomDataset
from taming.plotting_utils import save_image, save_image_grid, save_to_cvs
from taming.models.cond_transformer import Phylo_Net2NetTransformer

import torch
from tqdm import tqdm

import os

from omegaconf import OmegaConf
import argparse

from pathlib import Path

##########

@torch.no_grad()
def main(configs_yaml):
    yaml_paths = configs_yaml.yaml_path
    ckpt_paths = configs_yaml.ckpt_path
    DEVICE = configs_yaml.DEVICE
    file_list_path = configs_yaml.file_list_path    
    size = configs_yaml.size
    num_specimen_generated = configs_yaml.num_specimen_generated
    top_k = configs_yaml.top_k
    save_individual_images = configs_yaml.save_individual_images
    outputdir = configs_yaml.outputdir
    n_phylolevels = configs_yaml.n_phylolevels
    
    Path(outputdir).mkdir(parents=True, exist_ok=False)

    # load dataset
    dataset = CustomDataset(size, file_list_path, add_labels=True)
    indices_ = torch.tensor(range(len(dataset.indx_to_label)) )

    # nonphylo model
    non_phylo_model = load_phylovqvae(load_config(yaml_paths[-1], display=False), ckpt_path=ckpt_paths[-1], cuda=(DEVICE is not None), model_type=Phylo_Net2NetTransformer)
    
    prefixes = []
    previous_mapper = None
    for level in range(n_phylolevels):        
        # Load model
        model = load_phylovqvae(load_config(yaml_paths[level], display=False), ckpt_path=ckpt_paths[level], cuda=(DEVICE is not None), model_type=Phylo_Net2NetTransformer)
        codebooks_per_phylolevel = model.first_stage_model.phylo_disentangler.codebooks_per_phylolevel
        n_levels_non_attribute = model.first_stage_model.phylo_disentangler.n_levels_non_attribute
        
        # map indices
        indices = copy.copy(indices_)
        if model.cond_stage_model.phylo_mapper is not None:
            indices = sorted(list(set(model.cond_stage_model.phylo_mapper.get_mapped_truth(indices).tolist())))
        else:
            indices = sorted(list(set(indices.tolist())))
        
        prefixes_c = []      
        for c in indices:
            # get condition
            cond = torch.full((num_specimen_generated, 1), c).long().to(model.device)
            _, cond = model.encode_to_c(cond)
            
            # get prefix
            if level > 0:
                if level < 3:
                    c_species = model.cond_stage_model.phylo_mapper.get_reverse_indexing([c])[0] 
                else:
                    c_species = c
                c_adjusted = previous_mapper.get_mapped_truth(torch.LongTensor([c_species])).item()
                prefix = prefixes[level-1][c_adjusted] # should have n examples
            else:
                c_species = model.cond_stage_model.phylo_mapper.get_reverse_indexing([c])[0] 
                prefix = torch.zeros([num_specimen_generated, codebooks_per_phylolevel, 0], dtype=torch.long)
            prefix = prefix.to(model.device)
            
            # update condition to add prefix
            prefix_reshaped = model.first_stage_model.phylo_disentangler.embedding_converter.reshape_code(prefix)
            cond = torch.cat([cond, prefix_reshaped], dim=-1)
            
            
            #get postfix
            empty = torch.zeros([num_specimen_generated, 0], dtype=torch.long).to(model.device)
            e_l = model.sample(empty, cond, (n_phylolevels - level)*codebooks_per_phylolevel, sample=True, top_k=top_k) #(k, 24)
            z_l = model.first_stage_model.phylo_disentangler.embedding_converter.reshape_code(e_l, reverse=True) #(k, 8, 3)
            
            #aggregate phylo tokens
            phylo_e = torch.cat([prefix, z_l], dim=-1) #(k, 8, 4)
            phylo_e = model.first_stage_model.phylo_disentangler.embedding_converter.reshape_code(phylo_e) #(k, 32)
            z_phylo = model.first_stage_model.phylo_disentangler.embedding_converter.get_phylo_embeddings(phylo_e) #(k, 16, 8, 4)
            
            # get nonphylo tokens
            cond_nonphylo = torch.full((num_specimen_generated, 1), c).long().to(non_phylo_model.device)
            _, cond_nonphylo = non_phylo_model.encode_to_c(cond_nonphylo)
            cond_nonphylo = torch.cat([cond_nonphylo, phylo_e], dim=-1)
            e_non = non_phylo_model.sample(empty, cond_nonphylo, n_levels_non_attribute*codebooks_per_phylolevel, sample=True, top_k=top_k) #(k, 32)
            z_nonphylo = model.first_stage_model.phylo_disentangler.embedding_converter.get_phylo_embeddings(e_non) #(k, 16, 8, 4)
            
            #aggregate tokens 
            quant_z = torch.cat((z_phylo,z_nonphylo), dim=3) #(k, 16, 8, 8)
            total_e = torch.cat([phylo_e, e_non], dim=1) #(k, 64)
            
            # get image
            img = model.decode_to_img(total_e, quant_z.shape)
            log_image(img, dataset.indx_to_label[c_species], level, phylo_e, e_non, outputdir, save_individual_images=save_individual_images)
            
            #update prefixes for next level
            new_prefix = model.first_stage_model.phylo_disentangler.embedding_converter.get_sub_level(phylo_e.cpu(), level).view(num_specimen_generated, codebooks_per_phylolevel, -1)
            prefixes_c.append(new_prefix)
            
        previous_mapper = model.cond_stage_model.phylo_mapper
        
        prefixes.append(prefixes_c)

def log_image(img, lbl, level, phylo_e, e_non, outputdir, save_individual_images=False):
    postfix = "ancestor_level_{}".format(level)
    dir_ = os.path.join(outputdir, postfix)
    Path(dir_).mkdir(parents=True, exist_ok=True)
    
    if save_individual_images:
        for j in tqdm(range(img.shape[0])):   
            dir_species =       os.path.join(dir_,"{}".format(lbl))   
            Path(dir_species).mkdir(parents=True, exist_ok=True)        
            save_image(img[j, :, :, :], str(j), subfolder= dir_species)
    
    # Save the sequences
    save_to_cvs(None, dir_, "attributecodes_{}.csv".format(lbl), phylo_e.cpu().detach().numpy())
    save_to_cvs(None, dir_, "non_attributecodes_{}.csv".format(lbl), e_non.cpu().detach().numpy())

    # save the images
    save_image_grid(img, None, subfolder=dir_, postfix= "grid_{}".format(lbl))



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-n",
        "--config",
        type=str,
        nargs="?",
        const=True,
        default="analysis/configs/generate_with_transformer_idea4.yaml",
    )
    
    cfg, _ = parser.parse_known_args()
    configs = OmegaConf.load(cfg.config)
    cli = OmegaConf.from_cli()
    config = OmegaConf.merge(configs, cli)
    print(config)
    
    main(config)