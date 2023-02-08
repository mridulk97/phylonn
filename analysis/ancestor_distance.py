import argparse
import os
import pandas as pd
from scripts.data.custom import CustomTest as CustomDataset

from omegaconf import OmegaConf
from scripts.plotting_utils import plot_heatmap
import torch
import numpy as np

replace_nan = -1

def main(configs_yaml):
    fileNames = configs_yaml.fileNames
    size= configs_yaml.size
    file_list_path= configs_yaml.file_list_path
    
    dataset = CustomDataset(size, file_list_path, add_labels=True)
    
    # load csv
    for j, fileName in enumerate(fileNames):
        df = pd.read_csv(fileName)
        df = df.set_index('ml.names')
        print(df)
        
        list_of_species = list(dataset.labels_to_idx.keys())
        num_of_species = len(list_of_species)
        distances = torch.zeros([num_of_species, num_of_species])
        for indx, species1 in enumerate(list_of_species):
            if species1 in df.index:
                v1 = df.loc[species1]
                for indx2, species2 in enumerate(list_of_species[indx:]):
                    if species2 in df.index:
                        v2 = df.loc[species2]
                        distances[indx, indx2+indx] = distances[indx2+indx, indx] = np.linalg.norm(v1 - v2)
                    else:
                        distances[indx, indx2+indx] = distances[indx2+indx, indx] = replace_nan
                    
                        
            else:
                distances[indx, :] = replace_nan
                distances[:, indx] = replace_nan
                
        m = np.amax(distances.numpy())
        distances[distances == replace_nan] = torch.Tensor([m])
        
        plot_heatmap(distances, title="ancestor_distance_level{}".format(j), postfix=os.path.realpath(os.path.dirname(__file__)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-n",
        "--config",
        type=str,
        nargs="?",
        const=True,
        default="analysis/configs/ancestor_distance.yaml",
    )
    
    cfg, _ = parser.parse_known_args()
    configs = OmegaConf.load(cfg.config)
    cli = OmegaConf.from_cli()
    config = OmegaConf.merge(configs, cli)
    print(config)
    
    main(config)
        