from omegaconf import OmegaConf
import argparse
from taming.data.custom import CustomTest as CustomDataset
from PIL import Image
import torch
import os
from PIL import ImageFont
from PIL import ImageDraw 

from taming.modules.losses.phyloloss import parse_phyloDistances, get_relative_distance_for_level
from taming.data.phylogeny import Phylogeny

@torch.no_grad()
def main(configs_yaml):
    size = configs_yaml.size
    file_list_path = configs_yaml.file_list_path
    num_of_images_per_row = configs_yaml.num_of_images_per_row
    phylogeny_path = configs_yaml.phylogeny_path
    phyloDistances_string = configs_yaml.phyloDistances_string
    
    cols = num_of_images_per_row
    
    # load image
    dataset = CustomDataset(size, file_list_path, add_labels=True)
    
    paths = dataset.data.labels["file_path_"]
    labels = dataset.data.labels["class"]
    
    rows = len(set(labels))
    
    phylogeny = Phylogeny(phylogeny_path)
    phylo_distances = parse_phyloDistances(phyloDistances_string)
    for indx, i in enumerate(phylo_distances):
        phylo_distances[indx] = get_relative_distance_for_level(phylo_distances, indx)
    
    indices = {}
    for j in range(max(labels)+1):
        indices[j] = [i for i, x in enumerate(labels) if x == j]
        if len(indices[j]) > 4:
            indices[j] = indices[j][:4]
    
    
    grid = Image.new('RGB', size=(cols*size, rows*size))
    
    
    # print(indices)
    # print(paths)
    # print(labels)
    # print(grid)
    
    for j in range(max(labels)+1):
        species_name = dataset.indx_to_label[j]
        
        for indx, k in enumerate(indices[j]):
            im = Image.open(paths[k])
            draw = ImageDraw.Draw(im)
            
            txt = str(j) + "-" + species_name
            for d in phylo_distances:
                txt = txt + "-" + str(dataset.labels_to_idx[phylogeny.get_siblings_by_name(species_name, d)[0]])
            draw.text((0, 0), txt ,(0,0,0))

            grid.paste(im, box=(indx*size, j*size))
            
    grid.save(os.path.realpath(os.path.dirname(__file__))+"/species.pallette.png")
    
    
    
    
    
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-n",
        "--config",
        type=str,
        nargs="?",
        const=True,
        default="analysis/configs/generate_species_pallette.yaml",
    )
    
    cfg, _ = parser.parse_known_args()
    # cfg = parser.config
    configs = OmegaConf.load(cfg.config)
    cli = OmegaConf.from_cli()
    config = OmegaConf.merge(configs, cli)
    print(config)
    
    main(config)