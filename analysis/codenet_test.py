# License: BSD
# Author: Sasank Chilamkurthy


import os
from taming.analysis_utils import get_phylomapper_from_config
from taming.constants import QUANTIZED_PHYLO_OUTPUT
from taming.data.phylogeny import Phylogeny
from taming.loading_utils import load_config, load_phylovqvae
from taming.models.phyloautoencoder import PhyloVQVAE
from taming.plotting_utils import dump_to_json
import torch
import torch.nn as nn
# import torch.optim as optim
# from torch.optim import lr_scheduler
# import time
# import copy

from taming.data.custom import CustomTest #CustomTrain,

import argparse
from omegaconf import OmegaConf


def get_input(batch, k):
    x = batch[k]
    if len(x.shape) == 3:
        x = x[..., None]
    x = x.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format)
    return x.float()



def predict_model(model, encoding_model, level, dataloader, criterion, device, phylomapper=None):
    dataset_size = len(dataloader.dataset)
    
    encoding_model.eval()
    model.eval()   # Set model to evaluate mode

    running_loss = 0.0
    running_corrects = 0

    # Iterate over data.
    for x in dataloader:
        inputs = get_input(x, 'image').to(device)
        with torch.no_grad():
            _, _, _, in_out_disentangler = encoding_model(inputs)
            encoding = in_out_disentangler[QUANTIZED_PHYLO_OUTPUT]
            encoding = encoding_model.phylo_disentangler.embedding_converter.get_phylo_codes(encoding)
            encoding = encoding[:, :encoding_model.phylo_disentangler.n_phylolevels*encoding_model.phylo_disentangler.codebooks_per_phylolevel]
            encoding = encoding_model.phylo_disentangler.embedding_converter.get_sub_level(encoding, level).type(torch.FloatTensor).to(device) 
        labels = x['class'].to(device)

        with torch.set_grad_enabled(False):
            outputs = model(encoding)
            _, preds = torch.max(outputs, 1)
            
            if phylomapper is not None:
                labels = phylomapper.get_mapped_truth(labels)
            
            loss = criterion(outputs, labels)
            
        # statistics
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)


    loss = running_loss / dataset_size
    acc = running_corrects.double() / dataset_size

    print(f'Loss: {loss:.4f} Acc: {acc:.4f}')
    
    return acc



def get_hidden_layer_sizes(num_of_inputs, num_of_outputs, num_of_layers): 
    out_sizes = []
    if num_of_layers > 1:
        diff = num_of_inputs-num_of_outputs
        for i in range(num_of_layers-1):
            num_of_hidden = int(num_of_inputs - (i+1)*diff/num_of_layers)
            out_sizes.append(num_of_hidden)
    out_sizes.append(num_of_outputs)
    return out_sizes

# def make_MLP(input_dim, output_dim, num_of_layers = 1, normalize=False):        
#         out_sizes = get_hidden_layer_sizes(input_dim, output_dim, num_of_layers)
        
#         l =[]
#         if normalize:
#             l.append(nn.BatchNorm1d(input_dim))

#         in_ = input_dim 
#         for i in range(num_of_layers):
#             l = l + [nn.Linear(in_, out_sizes[i]),
#                 nn.SiLU(),
#             ]
#             in_ = out_sizes[i]
            
#         # Remove RELU from last layer
#         l = l[:-1]
        
#         return torch.nn.Sequential(*l)

def main(configs_yaml):
    DEVICE= configs_yaml.DEVICE
    size= configs_yaml.size
    postfix= configs_yaml.postfix
    file_list_path_test= configs_yaml.file_list_path_test
    batch_size= configs_yaml.batch_size
    num_workers= configs_yaml.num_workers
    encoding_model_path = configs_yaml.encoding_model_path
    phylogeny_path = configs_yaml.phylogeny_path
    encoding_model_config = configs_yaml.encoding_model_config
    # num_of_layers = configs_yaml.num_of_layers
    model_path = configs_yaml.model_path

    dataset_test = CustomTest(size, file_list_path_test, add_labels=True)
    
    dataloader = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size,
                                             shuffle=True, num_workers=num_workers)
    
    
    encoding_model_config = load_config(encoding_model_config, display=False)
    encoding_model = load_phylovqvae(encoding_model_config, ckpt_path=encoding_model_path, cuda=(DEVICE is not None), model_type=PhyloVQVAE)
    

    phylomapper = None
    if phylogeny_path is not None:
        phyloDistances_string = configs_yaml.phyloDistances_string
        level = configs_yaml.level
        
        phylomapper = get_phylomapper_from_config(Phylogeny(phylogeny_path), phyloDistances_string, level)

        # outputsize = phylomapper.get_len()
        
    else:
        level = 3
        # outputsize = len(dataset_test.indx_to_label.keys())
          
    # num_inputs = encoding_model.phylo_disentangler.codebooks_per_phylolevel*(level+1)

    # model_ft = make_MLP(num_inputs, outputsize, num_of_layers = num_of_layers, normalize=True)
    # print(model_ft)
    model_ft = torch.load(model_path)
    model_ft = model_ft.to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    
    
    acc = predict_model(model_ft, encoding_model, level, dataloader, criterion, DEVICE, phylomapper=phylomapper)
    
    save_path = os.path.dirname(encoding_model_path)
    torch.save(model_ft, os.path.join(save_path,"codenet_model_{}.path".format(level)))
    dump_to_json(acc, save_path, name='f1_Codenet- level {} {}'.format(level, postfix), get_fig_path=False)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-n",
        "--config",
        type=str,
        nargs="?",
        const=True,
        default="analysis/configs/codenet_test.yaml",
    )
    
    cfg, _ = parser.parse_known_args()
    # cfg = parser.config
    configs = OmegaConf.load(cfg.config)
    cli = OmegaConf.from_cli()
    config = OmegaConf.merge(configs, cli)
    print(config)
    
    main(config)