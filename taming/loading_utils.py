
import yaml
from taming.models.phyloautoencoder import PhyloVQVAE
from omegaconf import OmegaConf
import torch

######### loaders

def load_config(config_path, display=False):
    config = OmegaConf.load(config_path)
    if display:
        print(yaml.dump(OmegaConf.to_container(config)))
    return config


def load_phylovqvae(config, ckpt_path=None):
    model = PhyloVQVAE(**config.model.params)
    if ckpt_path is not None:
        sd = torch.load(ckpt_path, map_location="cpu")["state_dict"]
        missing, unexpected = model.load_state_dict(sd, strict=True)
    return model.eval()


