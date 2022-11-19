from taming.loading_utils import load_config, load_phylovqvae
from taming.data.custom import CustomTest as CustomDataset
from taming.data.utils import custom_collate
from taming.analysis_utils import Embedding_Code_converter, HistogramFrequency
from taming.models.vqgan import VQModel
from taming.plotting_utils import get_fig_pth

from torch.utils.data import DataLoader
import torch
from tqdm import tqdm

import taming.constants as CONSTANTS

import os

from omegaconf import OmegaConf
import argparse

GENERATED_FOLDER = "most_likely_generations"
GENERATED_DATASET = "generated_dataset"

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

    # load image
    dataset = CustomDataset(size, file_list_path, add_labels=True)
    dataloader = DataLoader(dataset.data, batch_size=batch_size, num_workers=num_workers, collate_fn=custom_collate)
    
    # Load model
    config = load_config(yaml_path, display=False)
    model = load_phylovqvae(config, ckpt_path=ckpt_path, data=dataset.data, cuda=(DEVICE is not None), model_type=VQModel)
        
    # create the converter.
    item = next(iter(dataloader))
    img = model.get_input(item, model.image_key).to(DEVICE)
    lbl = item[CONSTANTS.DISENTANGLER_CLASS_OUTPUT]
    q_phylo_output = model.encode(img)[0]
    
    converter_phylo = Embedding_Code_converter(model.quantize.get_codebook_entry_index, model.quantize.embedding, q_phylo_output[0, :, :, :].shape)
    
    q_phylo_output_indices = converter_phylo.get_phylo_codes(q_phylo_output[0, :, :, :].unsqueeze(0), verify=False)
    hist_freq = HistogramFrequency(len(dataset.indx_to_label.keys()), q_phylo_output_indices.shape[1])
        
    # collect values   
    hist_file_path = os.path.join(get_fig_pth(ckpt_path, postfix=CONSTANTS.HISTOGRAMS_FOLDER), CONSTANTS.HISTOGRAMS_FILE)
    try:
        hist_freq.load_from_file(hist_file_path) 
    except (OSError, IOError) as e:
        print(hist_file_path, '... Calculating histograms.')
        for item in tqdm(dataloader):
            img = model.get_input(item, model.image_key).to(DEVICE)
            lbl = item[CONSTANTS.DISENTANGLER_CLASS_OUTPUT]
            
            # get output
            q_phylo_output = model.encode(img)[0]
            # reshape 
            q_phylo_output_indices = converter_phylo.get_phylo_codes(q_phylo_output[0, :, :, :].unsqueeze(0), verify=False)
            
            hist_freq.set_location_frequencies(lbl, q_phylo_output_indices)

            
        hist_freq.save_to_file(hist_file_path)

         

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-n",
        "--config",
        type=str,
        nargs="?",
        const=True,
        default="analysis/configs/code_histogram_vqgan.yaml",
    )
    
    cfg, _ = parser.parse_known_args()
    # cfg = parser.config
    configs = OmegaConf.load(cfg.config)
    cli = OmegaConf.from_cli()
    config = OmegaConf.merge(configs, cli)
    print(config)
    
    main(config)