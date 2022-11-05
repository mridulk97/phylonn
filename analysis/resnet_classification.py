# License: BSD
# Author: Sasank Chilamkurthy


import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import models
import matplotlib.pyplot as plt
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



def train_model(model, dataloaders, criterion, optimizer, scheduler, device, num_epochs=25):
    since = time.time()
    
    dataset_sizes = {x: len(dataloaders[x].dataset) for x in ['train', 'val']}

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

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
                labels = x['class'].to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
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
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model




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
    
    
    dataset_train = CustomTrain(size, file_list_path_train, add_labels=True)
    dataset_test = CustomTest(size, file_list_path_test, add_labels=True)
    datasets = {
        'train': dataset_train, 
        'val': dataset_test
    }
    
    dataloaders = {x: torch.utils.data.DataLoader(datasets[x], batch_size=batch_size,
                                             shuffle=True, num_workers=num_workers)
              for x in ['train', 'val']}
    

    model_ft = models.resnet18(pretrained=True)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, len(dataset_train.indx_to_label.keys()))
    model_ft = model_ft.to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=lr, momentum=0.9)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=step_size, gamma=0.1)
    model_ft = train_model(model_ft, dataloaders, criterion, optimizer_ft, exp_lr_scheduler, DEVICE,
                        num_epochs=num_epochs)
    
    torch.save(model_ft, os.path.realpath(os.path.dirname(__file__))+"/classification_model.path")




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-n",
        "--config",
        type=str,
        nargs="?",
        const=True,
        default="analysis/configs/resnet_classification.yaml",
    )
    
    cfg, _ = parser.parse_known_args()
    # cfg = parser.config
    configs = OmegaConf.load(cfg.config)
    cli = OmegaConf.from_cli()
    config = OmegaConf.merge(configs, cli)
    print(config)
    
    main(config)