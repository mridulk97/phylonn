

from taming.loading_utils import load_config, load_phylovqvae
from taming.plotting_utils import save_image_grid
from taming.data.custom import CustomTest as CustomDataset
from taming.data.utils import custom_collate
from taming.analysis_utils import Embedding_Code_converter

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import taming.constants as CONSTANTS

from omegaconf import OmegaConf
import argparse


####################

class Clone_manger():
    def __init__(self, cummulative, embedding, codes):
        self.cummulative = cummulative
        self.embedding = embedding

        # if not cummulative:
        if cummulative:
        #         self.clone = torch.clone(embedding)
        # else:
            self.clone_list = []
            for code_index in codes:
                self.clone_list.append(torch.clone(embedding))

    def get_embedding(self, index=None):
        assert (index is not None) or (not self.cummulative)
        if not self.cummulative:
            return torch.clone(self.embedding)
        else:
            return self.clone_list[index]

@torch.no_grad()
def main(configs_yaml):
    yaml_path = configs_yaml.yaml_path
    ckpt_path = configs_yaml.ckpt_path
    DEVICE = configs_yaml.DEVICE
    imagepath = configs_yaml.imagepath
    batch_size = configs_yaml.batch_size
    file_list_path = configs_yaml.file_list_path
    num_workers = configs_yaml.num_workers
    size = configs_yaml.size
    cummulative = configs_yaml.cummulative
    plot_diff = configs_yaml.plot_diff

    # Load model
    config = load_config(yaml_path, display=False)
    model = load_phylovqvae(config, ckpt_path=ckpt_path).to(DEVICE)
    model.set_test_chkpt_path(ckpt_path)

    # load image
    dataset = CustomDataset(size, file_list_path, add_labels=True)
    dataloader = DataLoader(dataset.data, batch_size=batch_size, num_workers=num_workers, collate_fn=custom_collate)
    processed_img = torch.from_numpy(dataloader.dataset.preprocess_image(imagepath)).unsqueeze(0)
    processed_img = processed_img.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format)
    processed_img = processed_img.float()

    # get output
    dec_image, _, _, in_out_disentangler = model(processed_img.to(DEVICE))
    q_phylo_output = in_out_disentangler[CONSTANTS.QUANTIZED_PHYLO_OUTPUT]

    converter = Embedding_Code_converter(model.phylo_disentangler.quantize.get_codebook_entry_index, model.phylo_disentangler.quantize.embedding, q_phylo_output[0, :, :, :].shape)
    all_code_indices = converter.get_phylo_codes(q_phylo_output[0, :, :, :].unsqueeze(0), verify=False)
    all_codes_reverse_reshaped = converter.get_phylo_embeddings(all_code_indices, verify=False)

    dec_image_reversed, _, _, _ = model(processed_img.to(DEVICE), overriding_quant=all_codes_reverse_reshaped)
    
    
    which_codes = range(model.phylo_disentangler.n_embed)
    clone_manager = Clone_manger(cummulative, all_codes_reverse_reshaped, which_codes)

    for level in range(model.phylo_disentangler.n_phylolevels):
        for code_location in range(model.phylo_disentangler.codebooks_per_phylolevel):
            generated_imgs = [dec_image, dec_image_reversed]
                
            for code_index in tqdm(which_codes):
                all_codes_reverse_reshaped_clone = clone_manager.get_embedding(code_index)
                all_codes_reverse_reshaped_clone[0, :, code_location, level] = model.phylo_disentangler.quantize.embedding(torch.tensor([code_index]).to(all_codes_reverse_reshaped.device))

                dec_image_new, _, _, _ = model(processed_img.to(DEVICE), overriding_quant=all_codes_reverse_reshaped_clone)
                
                if plot_diff:
                    generated_imgs.append(dec_image_new - dec_image)
                else:
                    generated_imgs.append(dec_image_new)
            
            generated_imgs = torch.cat(generated_imgs, dim=0)
            save_image_grid(generated_imgs, ckpt_path, subfolder="codebook_grid-cumulative{}".format(cummulative), postfix="level{}-location{}".format(level, code_location))

        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-n",
        "--config",
        type=str,
        nargs="?",
        const=True,
        default="analysis/configs/replace_codes.yaml",
    )
    
    cfg, _ = parser.parse_known_args()
    # cfg = parser.config
    configs = OmegaConf.load(cfg.config)
    cli = OmegaConf.from_cli()
    config = OmegaConf.merge(configs, cli)
    print(config)
    
    main(config)