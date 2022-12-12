from taming.loading_utils import load_config, load_phylovqvae, load_CWVQGAN
from taming.modules.losses.phyloloss import get_loss_name
from taming.plotting_utils import save_image, save_image_grid, save_to_cvs
from taming.models.cond_transformer import Net2NetTransformer
from taming.data.custom import CustomTest as CustomDataset
from taming.analysis_utils import Embedding_Code_converter
from main import instantiate_from_config

import torch
from tqdm import tqdm



import os

from omegaconf import OmegaConf
import argparse

from torchmetrics import F1Score

GENERATED_FOLDER = "transformer_most_likely_generations"
GENERATED_DATASET = "transformer_generated_dataset"

# hist_arr structure:
#hist_arr[lbl][code_location] = [list of codes that occured]
#TODO: would be nice to have a class for this.

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
    model = load_CWVQGAN(config, ckpt_path=ckpt_path, data=dataset.data, cuda=(DEVICE is not None), model_type=Net2NetTransformer)
    indices = range(len(dataset.indx_to_label))
    # print(dataset.datasets["validation"].conditional_builders)
    # conditional_builder = dsets.datasets["validation"].conditional_builders[conditional]
    data = instantiate_from_config(config.data)
    data.prepare_data()
    data.setup()
    # if model.cond_stage_model.phylo_mapper is not None:
    #     indices = sorted(list(set(model.cond_stage_model.phylo_mapper.get_original_indexing_truth(indices))))
        # outputname = model.cond_stage_model.phylo_mapper.outputname
        # F1 = F1Score(num_classes=model.first_stage_model.phylo_disentangler.loss_phylo.classifier_output_sizes[model.cond_stage_model.level], multiclass=True) 
        
    print('generating images...')

    # For all species.
    for index, species_true_indx in enumerate(tqdm(indices)):  
        generate_images_org(index, species_true_indx,
                num_specimen_generated, top_k,
                model, dataset.indx_to_label,
                DEVICE, ckpt_path, outputdatasetdir, save_individual_images=save_individual_images)

def generate_images_org(index, species_true_indx, 
                    num_specimen_generated, top_k,
                    model, indx_to_label,
                    device, ckpt_path, prefix_text, save_individual_images=False):
    sequence_length = model.transformer.block_size-1
    
    
    list_of_created_sequence = []
    list_of_created_nonattribute_sequence = []

    # for all images
    generated_imgs = []
    steps = sequence_length
    c = torch.Tensor([index]).repeat(num_specimen_generated, 1).to(device).long()
    z_start = torch.zeros([num_specimen_generated, 0]).to(device).long()
    z_shape = (num_specimen_generated,
        model.first_stage_model.quantize.e_dim,  # codebook embed_dim
        16,  # z_height
        16  # z_width
    )


    # # generate the sequences. - det 
    # code_sampled = model.sample(z_start, c, steps=steps, sample=False)

    # generate the sequences. - topk 
    code_sampled = model.sample(z_start, c, steps=steps, sample=True, top_k=top_k)

    # decodeing the image from the sequence
    x_sample = model.decode_to_img(code_sampled, z_shape)

    for j in tqdm(range(num_specimen_generated)): 
        if save_individual_images:
            save_image(x_sample[j, :, :, :], str(j), ckpt_path, subfolder= os.path.join(GENERATED_DATASET,prefix_text,"{}".format(indx_to_label[species_true_indx])))

    return None


def generate_images(index, species_true_indx, 
                    num_specimen_generated, top_k,
                    model, indx_to_label,
                    device, ckpt_path, prefix_text, save_individual_images=False):
    
    sequence_length = model.transformer.block_size-1
    
    
    list_of_created_sequence = []
    list_of_created_nonattribute_sequence = []     
    
    
    
    #TODO: This whol file assumes we always have nonattr codes. We do not handle the case where there are only attr codes. We might want to either handle that properly or throw an error.
    assert (model.first_stage_model.phylo_disentangler.loss_anticlassification ), "This code only works with anticlassification for now."
    codebooks_per_phylolevel = model.first_stage_model.phylo_disentangler.codebooks_per_phylolevel
    n_phylolevels = model.first_stage_model.phylo_disentangler.n_phylolevels
    embed_dim = model.first_stage_model.phylo_disentangler.embed_dim
    n_levels_non_attribute = model.first_stage_model.phylo_disentangler.n_levels_non_attribute
    attr_codes_range = codebooks_per_phylolevel*n_phylolevels
    # nonattr_codes_range = codebooks_per_phylolevel*n_levels_non_attribute
        
    converter = Embedding_Code_converter(model.first_stage_model.phylo_disentangler.quantize.get_codebook_entry_index, model.first_stage_model.phylo_disentangler.quantize.embedding, (embed_dim, codebooks_per_phylolevel, n_phylolevels))
    converter_nonattribute = Embedding_Code_converter(model.first_stage_model.phylo_disentangler.quantize.get_codebook_entry_index, model.first_stage_model.phylo_disentangler.quantize.embedding, (embed_dim, codebooks_per_phylolevel, n_levels_non_attribute))
            
    # for all images
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
    
    for j in tqdm(range(num_specimen_generated)):    
        list_of_created_sequence.append(created_sequence[j, :].reshape(-1).tolist())
        list_of_created_nonattribute_sequence.append(created_nonattribute_sequence[j, :].reshape(-1).tolist())
        generated_imgs.append(dec_image_new[j, :, :, :].unsqueeze(0))
        
        if save_individual_images:
            save_image(dec_image_new[j, :, :, :], str(j), ckpt_path, subfolder= os.path.join(GENERATED_DATASET,prefix_text,"{}".format(indx_to_label[species_true_indx])))
        
    
    # Save the sequences
    save_to_cvs(ckpt_path, GENERATED_FOLDER, "{}_attributecodes_{}_{}.csv".format(prefix_text, index, indx_to_label[species_true_indx]), list_of_created_sequence)
    save_to_cvs(ckpt_path, GENERATED_FOLDER, "{}_non_attributecodes_{}_{}.csv".format(prefix_text, index, indx_to_label[species_true_indx]), list_of_created_nonattribute_sequence)

    # save the images
    generated_imgs = torch.cat(generated_imgs, dim=0)
    save_image_grid(generated_imgs, ckpt_path, subfolder=GENERATED_FOLDER, postfix= "{}_{}_{}".format(prefix_text, index, indx_to_label[species_true_indx]))



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
    # cfg = parser.config
    configs = OmegaConf.load(cfg.config)
    cli = OmegaConf.from_cli()
    config = OmegaConf.merge(configs, cli)
    print(config)
    
    main(config)