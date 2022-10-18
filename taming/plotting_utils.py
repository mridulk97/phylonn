import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path
import torchvision
import torch
import numpy as np
from PIL import Image
import json

def dump_to_json(dict, ckpt_path):
    root = get_fig_pth(ckpt_path)

    with open(os.path.join(root, "results.json"), "w") as outfile:
        json.dump(dict, outfile)

def save_image_grid(torch_images_4D, ckpt_path, subfolder=None, postfix="", nrow=10):
    root = get_fig_pth(ckpt_path, postfix=subfolder)

    grid = torchvision.utils.make_grid(torch_images_4D, nrow=nrow)
    grid = torch.clamp(grid, -1., 1.)

    grid = (grid+1.0)/2.0 # -1,1 -> 0,1; c,h,w
    grid = grid.transpose(0,1).transpose(1,2).squeeze(-1)
    grid = grid.cpu().numpy()
    grid = (grid*255).astype(np.uint8)
    filename = "code_changes"+postfix+".png"
    path = os.path.join(root, filename)
    os.makedirs(os.path.split(path)[0], exist_ok=True)
    Image.fromarray(grid).save(path)


def get_fig_pth(ckpt_path, postfix):
    postfix = os.path.join('figs', postfix)
    parent_path = Path(ckpt_path).parent.parent.absolute()
    fig_path = Path(os.path.join(parent_path, postfix))
    os.makedirs(fig_path, exist_ok=True)
    return fig_path

def plot_heatmap(heatmap, ckpt_path, title='default'):
    # show
    fig = plt.figure()
    ax = sns.heatmap(heatmap).set(title='heatmap of '+title)
    plt.show()
    fig.savefig(os.path.join(get_fig_pth(ckpt_path), title+ " heat_map.png"))