import argparse
import os
from taming.data.custom import CustomTest as CustomDataset
from omegaconf import OmegaConf
import torch
from collections import Counter
from tqdm import tqdm
import matplotlib.pyplot as plt

@torch.no_grad()
def main(configs_yaml):
    size = configs_yaml.size
    file_list_path_train = configs_yaml.file_list_path_train
    file_list_path_test = configs_yaml.file_list_path_test
    
    datasets = {
         'train': CustomDataset(size, file_list_path_train, add_labels=True),
         'test': CustomDataset(size, file_list_path_test, add_labels=True)
    }
    
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(40, 30))
    counters = {}
    k=0
    for key in datasets:
        dataset = datasets[key]
        classes_ = [batch['class'] for batch in tqdm(dataset.data)]
        counters[key] = Counter(classes_)
        axis_ticks = list(counters[key].keys())
        tick_counts = list(counters[key].values())
        axes[k].bar(axis_ticks, tick_counts)
        axes[k].set_title(key)
        axis_labels = [dataset.indx_to_label[int(i)] + "/"+ str(i) for i in counters[key].keys()]
        # print(axis_ticks, axis_labels, tick_counts)
        axes[k].set_xticks(axis_ticks, axis_labels, rotation=90)
        k=k+1
    
    fig.tight_layout()
    fig.savefig(os.path.realpath(os.path.dirname(__file__))+"/image_distribution.png")
       
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-n",
        "--config",
        type=str,
        nargs="?",
        const=True,
        default="analysis/configs/distribution_of_images.yaml",
    )
    
    cfg, _ = parser.parse_known_args()
    # cfg = parser.config
    configs = OmegaConf.load(cfg.config)
    cli = OmegaConf.from_cli()
    config = OmegaConf.merge(configs, cli)
    print(config)
    
    main(config)