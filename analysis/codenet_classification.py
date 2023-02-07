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
import torch.optim as optim
from torch.optim import lr_scheduler
import time
import copy

from taming.data.custom import CustomTrain, CustomTest

import argparse
from omegaconf import OmegaConf


def get_input(batch, k):
    x = batch[k]
    if len(x.shape) == 3:
        x = x[..., None]
    x = x.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format)
    return x.float()



def train_model(model, encoding_model, level, dataloaders, criterion, optimizer, scheduler, device, num_epochs=25, phylomapper=None):
    since = time.time()

    dataset_sizes = {x: len(dataloaders[x].dataset) for x in ['train', 'val']}

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = {
        'train': 0.0,
        'val': 0.0
    }
    
    encoding_model.eval()

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for x in dataloaders[phase]:
                inputs = get_input(x, 'image').to(device)
                with torch.no_grad():
                    _, _, _, in_out_disentangler = encoding_model(inputs)
                    encoding = in_out_disentangler[QUANTIZED_PHYLO_OUTPUT]
                    encoding = encoding_model.phylo_disentangler.embedding_converter.get_phylo_codes(encoding)
                    encoding = encoding[:, :encoding_model.phylo_disentangler.n_phylolevels*encoding_model.phylo_disentangler.codebooks_per_phylolevel]
                    encoding = encoding_model.phylo_disentangler.embedding_converter.get_sub_level(encoding, level).type(torch.FloatTensor).to(device) 
                    # encoding = encoding/encoding_model.phylo_disentangler.n_embed
                labels = x['class'].to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(encoding)
                    _, preds = torch.max(outputs, 1)
                    
                    if phylomapper is not None:
                        labels = phylomapper.get_mapped_truth(labels)
                    
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # deep copy the model
            if phase == 'train':
                train_acc = epoch_acc
            if phase == 'val' and epoch_acc.item() > best_acc['val']:
                best_acc['val'] = epoch_acc.item()
                best_acc['train'] = train_acc.item()
                best_model_wts = copy.deepcopy(model.state_dict())

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    best_val_acc = best_acc['val']
    print(f'Best val Acc: {best_val_acc:4f}')

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model , best_acc



def get_hidden_layer_sizes(num_of_inputs, num_of_outputs, num_of_layers): 
    out_sizes = []
    if num_of_layers > 1:
        diff = num_of_inputs-num_of_outputs
        for i in range(num_of_layers-1):
            num_of_hidden = int(num_of_inputs - (i+1)*diff/num_of_layers)
            out_sizes.append(num_of_hidden)
    out_sizes.append(num_of_outputs)
    return out_sizes

def make_MLP(input_dim, output_dim, num_of_layers = 1, normalize=False):        
        out_sizes = get_hidden_layer_sizes(input_dim, output_dim, num_of_layers)
        
        l =[]
        if normalize:
            l.append(nn.BatchNorm1d(input_dim))

        in_ = input_dim 
        for i in range(num_of_layers):
            l = l + [nn.Linear(in_, out_sizes[i]),
                nn.SiLU(),
            ]
            in_ = out_sizes[i]
            
        # Remove RELU from last layer
        l = l[:-1]
        
        return torch.nn.Sequential(*l)

def main(configs_yaml):
    DEVICE= configs_yaml.DEVICE
    size= configs_yaml.size
    file_list_path_train= configs_yaml.file_list_path_train
    file_list_path_test= configs_yaml.file_list_path_test
    batch_size= configs_yaml.batch_size
    num_workers= configs_yaml.num_workers
    step_size = configs_yaml.step_size
    lr = configs_yaml.lr
    num_epochs = configs_yaml.num_epochs
    encoding_model_path = configs_yaml.encoding_model_path
    phylogeny_path = configs_yaml.phylogeny_path
    encoding_model_config = configs_yaml.encoding_model_config
    num_of_layers = configs_yaml.num_of_layers

    dataset_train = CustomTrain(size, file_list_path_train, add_labels=True)
    dataset_test = CustomTest(size, file_list_path_test, add_labels=True)
    datasets = {
        'train': dataset_train, 
        'val': dataset_test
    }
    
    dataloaders = {x: torch.utils.data.DataLoader(datasets[x], batch_size=batch_size,
                                             shuffle=True, num_workers=num_workers)
              for x in ['train', 'val']}
    
    
    encoding_model_config = load_config(encoding_model_config, display=False)
    encoding_model = load_phylovqvae(encoding_model_config, ckpt_path=encoding_model_path, cuda=(DEVICE is not None), model_type=PhyloVQVAE)
    

    phylomapper = None
    if phylogeny_path is not None:
        phyloDistances_string = configs_yaml.phyloDistances_string
        level = configs_yaml.level
        
        phylomapper = get_phylomapper_from_config(Phylogeny(phylogeny_path), phyloDistances_string, level)

        outputsize = phylomapper.get_len()
        
    else:
        level = 3
        outputsize = len(dataset_train.indx_to_label.keys())
          
    num_inputs = encoding_model.phylo_disentangler.codebooks_per_phylolevel*(level+1)

    model_ft = make_MLP(num_inputs, outputsize, num_of_layers = num_of_layers, normalize=True)
    print(model_ft)
    model_ft = model_ft.to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=lr, momentum=0.9)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=step_size, gamma=1.0)
    
    
    model_ft, best_acc = train_model(model_ft, encoding_model, level, dataloaders, criterion, optimizer_ft, exp_lr_scheduler, DEVICE,
                        num_epochs=num_epochs, phylomapper=phylomapper)
    
    save_path = os.path.dirname(encoding_model_path)
    torch.save(model_ft, os.path.join(save_path,"codenet_model_{}.path".format(level)))
    dump_to_json(best_acc, save_path, name='f1_Codenet- level {} {}'.format(level, 'train.val'), get_fig_path=False)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-n",
        "--config",
        type=str,
        nargs="?",
        const=True,
        default="analysis/configs/codenet_classification.yaml",
    )
    
    cfg, _ = parser.parse_known_args()
    # cfg = parser.config
    configs = OmegaConf.load(cfg.config)
    cli = OmegaConf.from_cli()
    config = OmegaConf.merge(configs, cli)
    print(config)
    
    main(config)