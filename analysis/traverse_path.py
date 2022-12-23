
import os
from taming.analysis_utils import Embedding_Code_converter, get_phylomapper_from_config
from taming.data.phylogeny import Phylogeny
from taming.loading_utils import load_config, load_phylovqvae
from taming.data.custom import CustomTest as CustomDataset
from taming.models.cond_transformer import Phylo_Net2NetTransformer

import taming.constants as CONSTANTS
from taming.modules.losses.phyloloss import parse_phyloDistances
from taming.plotting_utils import save_image_grid

import torch

from omegaconf import OmegaConf
import argparse

class Model_loader:
    def __init__(self, transformer_paths):
        self.transformer_paths = transformer_paths
        self.models = [None]*len(transformer_paths)
        
    def load_model(self, model_index, DEVICE):
        if self.models[model_index] is None:
            config_species = self.transformer_paths[model_index]['yaml_path']
            ckpt_path_species = self.transformer_paths[model_index]['ckpt_path']
            config = load_config(config_species, display=False)
            model = load_phylovqvae(config, ckpt_path=ckpt_path_species, cuda=(DEVICE is not None), model_type=Phylo_Net2NetTransformer)
            self.models[model_index] = model
            
        return self.models[model_index]
    
@torch.no_grad()
def main(configs_yaml):
    transformer_paths = configs_yaml.transformer_paths
    DEVICE= configs_yaml.DEVICE
    top_k = configs_yaml.top_k
    size= configs_yaml.size
    file_list_path= configs_yaml.file_list_path
    image_index1= configs_yaml.image_index1
    image_index2= configs_yaml.image_index2
    phyloDistances_string = configs_yaml.phyloDistances_string
    phylogeny_path = configs_yaml.phylogeny_path
    num_of_samples = configs_yaml.num_of_samples
    
    phylo_distances = parse_phyloDistances(phyloDistances_string)
    
    # load image
    dataset = CustomDataset(size, file_list_path, add_labels=True)
    
    # Load model
    model_loader = Model_loader(transformer_paths)
    species_model = model_loader.load_model(len(phylo_distances), DEVICE).first_stage_model
    
    # load image. get class and zq_phylo and z_qnonattr
    classes = []
    codes = []
    dec_images = []
    converter_phylo = None
    converter_nonattr = None
    len_phylo = None
    imgs = []
    for img_indx in [image_index1, image_index2]:
        specimen = dataset[img_indx]
        
        processed_img = specimen['image']
        processed_img = torch.from_numpy(processed_img)
        processed_img = processed_img.unsqueeze(0)
        processed_img = processed_img.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format)
        processed_img = processed_img.float()
        processed_img = processed_img.to(DEVICE)
        
        dec_image, _, _, in_out_disentangler = species_model(processed_img)
        dec_images.append(dec_image)
        imgs.append(processed_img)
        
        
        q_phylo = in_out_disentangler[CONSTANTS.QUANTIZED_PHYLO_OUTPUT] 
        q_non_attr = in_out_disentangler[CONSTANTS.QUANTIZED_PHYLO_NONATTRIBUTE_OUTPUT] 

        if converter_phylo is None:
            converter_phylo = Embedding_Code_converter(species_model.phylo_disentangler.quantize.get_codebook_entry_index, species_model.phylo_disentangler.quantize.embedding, q_phylo[0, :, :, :].shape)
            converter_nonattr = Embedding_Code_converter(species_model.phylo_disentangler.quantize.get_codebook_entry_index, species_model.phylo_disentangler.quantize.embedding, q_non_attr[0, :, :, :].shape)
        
        all_code_indices_phylo = converter_phylo.get_phylo_codes(q_phylo[0, :, :, :].unsqueeze(0), verify=False)
        all_code_indices_nonattr = converter_nonattr.get_phylo_codes(q_non_attr[0, :, :, :].unsqueeze(0), verify=False)
        if len_phylo is None:
            len_phylo = all_code_indices_phylo.view(1, -1).shape[-1]
        
        all_code = torch.cat([all_code_indices_phylo, all_code_indices_nonattr], dim=1)
        codes.append(all_code)
        
        species_index = specimen['class']
        classes.append(species_index)
        print(specimen['file_path_'], specimen['class'])
        

    # Get ancestry sequence
    ancestors = []
    for l_indx, l in enumerate(phylo_distances):
        phylomapper = get_phylomapper_from_config(Phylogeny(phylogeny_path), phyloDistances_string, l_indx)
        
        ancestor = phylomapper.get_mapped_truth(torch.LongTensor(classes))
        ancestors.append(ancestor)
    
    # go up the tree
    total_ancestors = len(ancestors)
    last_index = total_ancestors+1
    generated_imgs = []
    for indx, i in enumerate(range(total_ancestors)):
        reversed_index = total_ancestors - 1 - indx
        last_index = reversed_index
        
        a1 = ancestors[reversed_index][0]
        a2 = ancestors[reversed_index][1]
        
        transformer = model_loader.load_model(reversed_index, DEVICE)
        original_code = codes[0].repeat(num_of_samples, 1)
        
        
        c = torch.LongTensor([a1]).repeat(num_of_samples, 1).to(DEVICE)
        z = torch.zeros([num_of_samples, 0]).to(DEVICE).long()
        generated_code = transformer.sample(z, c, original_code.shape[-1], sample=True, top_k=top_k)
        
        mixed_code, nonattr_generated_code = generate_mixed_image(reversed_index, original_code, generated_code, converter_phylo, len_phylo)
        
        dec_image_new = generate_image_from_codes(transformer, converter_phylo, converter_nonattr, mixed_code, nonattr_generated_code)
        
        generated_imgs.append(dec_image_new)
        
        if a1 == a2:
            print("reached common ancestor at level", reversed_index)
            break
        
    # Go down the tree
    if last_index < total_ancestors-1:
        for indx, i in enumerate(range(last_index+1, total_ancestors)):
            a2 = ancestors[indx][1]
                
            transformer = model_loader.load_model(indx, DEVICE)
            original_code = codes[1].repeat(num_of_samples, 1)
                    
            c = torch.LongTensor([a2]).repeat(num_of_samples, 1).to(DEVICE)
            z = torch.zeros([num_of_samples, 0]).to(DEVICE).long()
            generated_code = transformer.sample(z, c, original_code.shape[-1], sample=True, top_k=top_k)
            
            mixed_code, nonattr_generated_code = generate_mixed_image(reversed_index, original_code, generated_code, converter_phylo, len_phylo)
            
            dec_image_new = generate_image_from_codes(transformer, converter_phylo, converter_nonattr, mixed_code, nonattr_generated_code)
            
            generated_imgs.append(dec_image_new)
        
    #save images
    for i in range(num_of_samples):
        sub_generated_images = list(map(lambda x: x[i, :, :, :].unsqueeze(0), generated_imgs))
        conc_imgs = [imgs[0]]+ sub_generated_images + [imgs[1]]
        conc_imgs = torch.cat(conc_imgs, dim=0)
        
        save_image_grid(conc_imgs, postfix= "morphing_from_{}_to_{}_{}".format(image_index1, image_index2, i), subfolder=os.path.realpath(os.path.dirname(__file__)))
    

def generate_mixed_image(index, original_code, generated_code, converter_phylo, len_phylo):
    phylo_generated_code = generated_code[:, :len_phylo].view(original_code.shape[0], -1)
    nonattr_generated_code = generated_code[:, len_phylo:].view(original_code.shape[0], -1)
    phylo_original_code = original_code[:, :len_phylo].view(original_code.shape[0], -1)
    sub_original_code = converter_phylo.get_sub_level(phylo_original_code, index)
    mixed_code = converter_phylo.set_sub_level(phylo_generated_code, sub_original_code, index)
    
    return  mixed_code, nonattr_generated_code

def generate_image_from_codes(transformer, converter_phylo, converter_nonattr, mixed_code, nonattr_generated_code):
    embedding = converter_phylo.get_phylo_embeddings(mixed_code)
    embedding_nonattribute = converter_nonattr.get_phylo_embeddings(nonattr_generated_code)
    dec_image_new, _ = transformer.first_stage_model.from_quant_only(embedding, embedding_nonattribute)
    return dec_image_new
                
                
                
                
            
            
        
    
        
            
        
            
        
    
        
        
    
        
    
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-n",
        "--config",
        type=str,
        nargs="?",
        const=True,
        default="analysis/configs/traverse_path.yaml",
    )
    
    cfg, _ = parser.parse_known_args()
    # cfg = parser.config
    configs = OmegaConf.load(cfg.config)
    cli = OmegaConf.from_cli()
    config = OmegaConf.merge(configs, cli)
    print(config)
    
    main(config)