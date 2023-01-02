# License: BSD
# Author: Sasank Chilamkurthy


import os
from taming.analysis_utils import get_phylomapper_from_config
from taming.data.phylogeny import Phylogeny
import torch
import torch.nn as nn
import shutil

import argparse
from omegaconf import OmegaConf
from tqdm import tqdm


@torch.no_grad()
def main(configs_yaml):
    dataset= configs_yaml.dataset
    level= configs_yaml.level
    phylogeny_path = configs_yaml.phylogeny_path
    phyloDistances_string = configs_yaml.phyloDistances_string
    
    phylomapper = get_phylomapper_from_config(Phylogeny(phylogeny_path), phyloDistances_string, level)
    
    root, dirs, files = next(os.walk(dataset))
    
    new_dir = root+"_level"+str(level)
    
    
    sorted_dirs = list(sorted(dirs))
    for indx, dir in enumerate(sorted_dirs):
        tgt = phylomapper.get_original_indexing_truth([indx])
        tgt_name = sorted_dirs[tgt[0]]
        print(indx, dir, "->", tgt, tgt_name)
        target_path = os.path.join(new_dir,tgt_name)
        # os.makedirs(target_path, exist_ok=True)
        src = os.path.join(root,dir)
        print(src, target_path)
        shutil.copytree(src, target_path,dirs_exist_ok=True)
            
            
    
 
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-n",
        "--config",
        type=str,
        nargs="?",
        const=True,
        default="analysis/configs/create_ancestor_dataset.yaml",
    )
    
    cfg, _ = parser.parse_known_args()
    configs = OmegaConf.load(cfg.config)
    cli = OmegaConf.from_cli()
    config = OmegaConf.merge(configs, cli)
    print(config)
    
    main(config)