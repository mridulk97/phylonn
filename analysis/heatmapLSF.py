import argparse, os, sys, datetime, glob, importlib
from omegaconf import OmegaConf
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
from pytorch_lightning.trainer import Trainer

from tqdm import tqdm
from scripts.data.utils import custom_collate
from scripts.plotting_utils import get_fig_pth
from scripts.analysis_utils import aggregate_metric_from_specimen_to_species, get_CosineDistance_matrix
from scripts.plotting_utils import plot_heatmap

from main import WrappedDataset, DataModuleFromConfig
from scripts.import_utils import instantiate_from_config


def get_parser(**parser_kwargs):
    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument(
        "-b",
        "--config",
        default="configs/lsf_inference.yaml",
    )
    return parser


def distance_matrix(model, dataset, plot_save_dir):

    idx_to_label = {y:x for x, y in dataset.labels_to_idx.items()}

    encodings = []
    classes = []

    with torch.no_grad():
        for d in dataset:
            img = torch.tensor(d['image']).unsqueeze(0).permute(0, 3, 1, 2).cuda()
            z, _, _ = model.image2encoding(img)
            z = (z @ model.LSF_disentangler.M.t())[:, :38]
            encodings.append(z)
            classes.append(d['class'])

    encodings = torch.cat(encodings)

    sort_indices = np.argsort(classes)
    sorted_encodings = encodings[sort_indices,:]
    sorted_classes = sorted(classes)

    z_cosine_distances = get_CosineDistance_matrix(sorted_encodings)

    os.makedirs(plot_save_dir, exist_ok=True)

    plot_heatmap(z_cosine_distances.cpu(), postfix=plot_save_dir,
                    title='z cosine distances')

    sorted_class_names_according_to_class_indx = [idx_to_label[x] for x in sorted_classes]
    embedding_dist = aggregate_metric_from_specimen_to_species(sorted_class_names_according_to_class_indx,
                                                                z_cosine_distances)        
    plot_heatmap(embedding_dist.cpu(), postfix=plot_save_dir,
                    title='z cosine species distances')  


@torch.no_grad()
def main():

    now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")

    sys.path.append(os.getcwd())

    parser = get_parser()
    opt, unknown = parser.parse_known_args()
    config = OmegaConf.load(opt.config)

    # model
    model = instantiate_from_config(config.model)
    model = model.cuda()

    data = instantiate_from_config(config.data)
    data.prepare_data()
    data.setup()
    val_data = data.datasets['validation']    
    
    plot_save_dir = get_fig_pth(config.model.params.LSF_params.ckpt_path)
    plot_save_dir = os.path.join(str(plot_save_dir), "heat_map_cosine_LSF")
    distance_matrix(model, val_data, plot_save_dir)

if __name__ == "__main__":
    main()
