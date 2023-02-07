
import yaml
from taming.models.phyloautoencoder import PhyloVQVAE
from taming.models.cwautoencoder import CWmodelVQGAN
from taming.models.LSFautoencoder import LSFVQVAE
from omegaconf import OmegaConf
import torch

######### loaders

def load_config(config_path, display=False):
    config = OmegaConf.load(config_path)
    if display:
        print(yaml.dump(OmegaConf.to_container(config)))
    return config


#TODO: This should become one function for all models.
def load_phylovqvae(config, ckpt_path=None, cuda=False, model_type=PhyloVQVAE):
    model = model_type(**config.model.params)
    if ckpt_path is not None:
        sd = torch.load(ckpt_path, map_location="cpu")["state_dict"]
        missing, unexpected = model.load_state_dict(sd, strict=True)
    if cuda:
        model = model.cuda()
    return model.eval()

def load_CWVQGAN(config, ckpt_path=None, data=None, cuda=False, model_type=CWmodelVQGAN):
    model = model_type(**config.model.params)
    if ckpt_path is not None:
        sd = torch.load(ckpt_path, map_location="cpu")["state_dict"]
        missing, unexpected = model.load_state_dict(sd, strict=True)
    if cuda:
        model = model.cuda()
    return model.eval()

def load_LSFvqvae(config, ckpt_path=None, data=None, cuda=False, model_type=LSFVQVAE):
    model = model_type(**config.model.params)
    if ckpt_path is not None:
        print('Loading model from', ckpt_path)
        sd = torch.load(ckpt_path, map_location="cpu")["state_dict"]
        missing, unexpected = model.load_state_dict(sd, strict=False)
    if cuda:
        model = model.cuda()
    return model.eval()


