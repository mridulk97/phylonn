from scripts.loading_utils import load_config, load_CWVQGAN
from scripts.models.cond_transformer import Net2NetTransformer
from scripts.data.custom import CustomTest as CustomDataset
from main import instantiate_from_config

import torch
from tqdm import tqdm

from omegaconf import OmegaConf
import argparse

##########

@torch.no_grad()
def main_cw(configs_yaml):
    yaml_path = configs_yaml.yaml_path
    ckpt_path = configs_yaml.ckpt_path
    DEVICE = configs_yaml.DEVICE
    file_list_path = configs_yaml.file_list_path    
    size = configs_yaml.size
    num_specimen_generated = configs_yaml.num_specimen_generated
    top_k = configs_yaml.top_k
    outputdatasetdir = configs_yaml.outputdatasetdir

    # load image
    dataset = CustomDataset(size, file_list_path, add_labels=True)

    # Load model
    config = load_config(yaml_path, display=False)
    model = load_CWVQGAN(config, ckpt_path=ckpt_path, data=dataset.data, cuda=(DEVICE is not None), model_type=Net2NetTransformer)
    indices = range(len(dataset.indx_to_label))
    data = instantiate_from_config(config.data)
    data.prepare_data()
    data.setup()
        
    print('generating images...')

    # For all species.
    for index, species_true_indx in enumerate(tqdm(indices)):  
        generate_images_cw(index, species_true_indx,
                num_specimen_generated, top_k,
                model, dataset.indx_to_label,
                DEVICE, ckpt_path, outputdatasetdir)


def generate_images_cw(index, species_true_indx, 
                    num_specimen_generated, top_k,
                    model, indx_to_label,
                    device, ckpt_path, prefix_text):
    sequence_length = model.transformer.block_size-1

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

    # generate the sequences. - topk 
    code_sampled = model.sample(z_start, c, steps=steps, sample=True, top_k=top_k)

    # decodeing the image from the sequence
    x_sample = model.decode_to_img(code_sampled, z_shape) #TODO: Mridul fix this.
    return None



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

    main_cw(config)