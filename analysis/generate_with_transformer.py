from scripts.loading_utils import load_config, load_model
from scripts.data.custom import CustomTest as CustomDataset
from scripts.analysis_utils import Embedding_Code_converter
from scripts.plotting_utils import save_image, save_image_grid, save_to_cvs
from scripts.models.cond_transformer import PhyloNN_transformer

import torch
from tqdm import tqdm

import os

from omegaconf import OmegaConf
import argparse

GENERATED_FOLDER = "transformer_most_likely_generations"
GENERATED_DATASET = "transformer_generated_dataset"

##########

@torch.no_grad()
def main(configs_yaml):
    yaml_path = configs_yaml.yaml_path
    ckpt_path = configs_yaml.ckpt_path
    DEVICE = configs_yaml.DEVICE
    file_list_path = configs_yaml.file_list_path    
    size = configs_yaml.size
    num_specimen_generated = configs_yaml.num_specimen_generated
    top_k = configs_yaml.top_k
    save_individual_images = configs_yaml.save_individual_images
    outputdatasetdir = configs_yaml.outputdatasetdir

    # load image
    dataset = CustomDataset(size, file_list_path, add_labels=True)

    # Load model
    config = load_config(yaml_path, display=False)
    model = load_model(config, ckpt_path=ckpt_path, cuda=(DEVICE is not None), model_type=PhyloNN_transformer)
    
    # generate the images
    if not model.be_unconditional:
        indices = range(len(dataset.indx_to_label))
        if model.cond_stage_model.phylo_mapper is not None:
            indices = sorted(list(set(model.cond_stage_model.phylo_mapper.get_original_indexing_truth(indices))))

        print('generating images...')

        # For all species.
        for index, species_true_indx in enumerate(tqdm(indices)):  
            lbl = dataset.indx_to_label[species_true_indx]
            generate_images(index, lbl,
                    num_specimen_generated, top_k,
                    model,
                    DEVICE, ckpt_path, outputdatasetdir, save_individual_images=save_individual_images)
    else:
        generate_images(0, "unconditional",
                    num_specimen_generated, top_k,
                    model,
                    DEVICE, ckpt_path, outputdatasetdir, save_individual_images=save_individual_images)


def generate_images(index, lbl, 
                    num_specimen_generated, top_k,
                    model,
                    device, ckpt_path, prefix_text, save_individual_images=False):
    
    sequence_length = model.transformer.get_block_size()-1
    
    list_of_created_sequence = []
    list_of_created_nonattribute_sequence = []     
    
    codes_per_phylolevel = model.first_stage_model.phylo_disentangler.codes_per_phylolevel
    n_phylolevels = model.first_stage_model.phylo_disentangler.n_phylolevels
    embed_dim = model.first_stage_model.phylo_disentangler.embed_dim
    n_levels_non_attribute = model.first_stage_model.phylo_disentangler.n_levels_non_attribute
    attr_codes_range = codes_per_phylolevel*n_phylolevels
        
    converter = Embedding_Code_converter(model.first_stage_model.phylo_disentangler.quantize.get_codebook_entry_index, model.first_stage_model.phylo_disentangler.quantize.embedding, (embed_dim, codes_per_phylolevel, n_phylolevels))
    converter_nonattribute = Embedding_Code_converter(model.first_stage_model.phylo_disentangler.quantize.get_codebook_entry_index, model.first_stage_model.phylo_disentangler.quantize.embedding, (embed_dim, codes_per_phylolevel, n_levels_non_attribute))

    # construct the label conditioning
    generated_imgs = []
    steps = sequence_length
    c = torch.Tensor([index]).repeat(num_specimen_generated, 1).to(device).long()
    
    # generate the sequences.
    code = model.sample(torch.zeros([num_specimen_generated, 0]).to(device).long(), c, steps, sample=True, top_k=top_k)
            
    # append the generated sequence
    created_nonattribute_sequence = code[:, attr_codes_range:].view(num_specimen_generated, -1)
    created_sequence = code[:, :attr_codes_range].view(num_specimen_generated, -1)
    embedding = converter.get_phylo_embeddings(created_sequence)
    embedding_nonattribute = converter_nonattribute.get_phylo_embeddings(created_nonattribute_sequence)
    dec_image_new, _ = model.first_stage_model.from_quant_only(embedding, embedding_nonattribute)
    
    # save the images
    for j in tqdm(range(num_specimen_generated)):    
        list_of_created_sequence.append(created_sequence[j, :].reshape(-1).tolist())
        list_of_created_nonattribute_sequence.append(created_nonattribute_sequence[j, :].reshape(-1).tolist())
        generated_imgs.append(dec_image_new[j, :, :, :].unsqueeze(0))
        
        if save_individual_images:
            save_image(dec_image_new[j, :, :, :], str(j), ckpt_path, subfolder= os.path.join(GENERATED_DATASET,prefix_text,"{}".format(lbl)))
        
    
    # Save the sequences
    save_to_cvs(ckpt_path, GENERATED_FOLDER, "{}_attributecodes_{}_{}.csv".format(prefix_text, index, lbl), list_of_created_sequence)
    save_to_cvs(ckpt_path, GENERATED_FOLDER, "{}_non_attributecodes_{}_{}.csv".format(prefix_text, index, lbl), list_of_created_nonattribute_sequence)

    # save the images as a grid
    generated_imgs = torch.cat(generated_imgs, dim=0)
    save_image_grid(generated_imgs, ckpt_path, subfolder=GENERATED_FOLDER, postfix= "{}_{}_{}".format(prefix_text, index, lbl))



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-n",
        "--config",
        type=str,
        nargs="?",
        const=True,
        default="analysis/configs/generate_with_transformer.yaml",
    )
    
    cfg, _ = parser.parse_known_args()
    configs = OmegaConf.load(cfg.config)
    cli = OmegaConf.from_cli()
    config = OmegaConf.merge(configs, cli)
    print(config)
    
    main(config)