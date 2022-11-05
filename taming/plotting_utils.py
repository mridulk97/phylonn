import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path
import torchvision
import torch
import numpy as np
from PIL import Image
import json
import csv

from sklearn.metrics import ConfusionMatrixDisplay

def dump_to_json(dict, ckpt_path, name='results'):
    root = get_fig_pth(ckpt_path)

    with open(os.path.join(root, name+".json"), "w") as outfile:
        json.dump(dict, outfile)
        

def save_to_cvs(ckpt_path, postfix, file_name, list_of_created_sequence):
    file = open(os.path.join(get_fig_pth(ckpt_path, postfix=postfix), file_name), 'w')
    with file:  
        write = csv.writer(file)
        write.writerows(list_of_created_sequence)
        
def save_to_txt(arr, ckpt_path, name='results'):
    root = get_fig_pth(ckpt_path)
    with open(os.path.join(root, name+".txt"), "w") as outfile:
        outfile.write(str(arr))



def save_image_grid(torch_images_4D, ckpt_path, subfolder=None, postfix="", nrow=10):
    root = get_fig_pth(ckpt_path, postfix=subfolder)

    grid = torchvision.utils.make_grid(torch_images_4D, nrow=nrow)
    grid = torch.clamp(grid, -1., 1.)

    grid = (grid+1.0)/2.0 # -1,1 -> 0,1; c,h,w
    grid = grid.transpose(0,1).transpose(1,2).squeeze(-1)
    grid = grid.cpu().numpy()
    grid = (grid*255).astype(np.uint8)
    filename = "code_changes_"+postfix+".png"
    path = os.path.join(root, filename)
    os.makedirs(os.path.split(path)[0], exist_ok=True)
    Image.fromarray(grid).save(path)


def get_fig_pth(ckpt_path, postfix=None):
    figs_postfix = 'figs'
    postfix = os.path.join(figs_postfix, postfix) if postfix is not None else figs_postfix
    parent_path = Path(ckpt_path).parent.parent.absolute()
    fig_path = Path(os.path.join(parent_path, postfix))
    os.makedirs(fig_path, exist_ok=True)
    return fig_path

def plot_heatmap(heatmap, ckpt_path, title='default', postfix=None):
    # show
    fig = plt.figure()
    ax = sns.heatmap(heatmap).set(title='heatmap of '+title)
    plt.show()
    fig.savefig(os.path.join(get_fig_pth(ckpt_path, postfix=postfix), title+ " heat_map.png"))



def plot_confusionmatrix(preds, classes, classnames, ckpt_path, postfix=None, title=""):
    fig, ax = plt.subplots(figsize=(30,30))
    preds_max = np.argmax(preds.cpu().numpy(), axis=-1)
    disp = ConfusionMatrixDisplay.from_predictions(classes.cpu().numpy(), preds_max, display_labels=classnames, normalize='true', xticks_rotation='vertical', ax=ax)
    disp.plot()
    fig.savefig(os.path.join(get_fig_pth(ckpt_path, postfix=postfix), title+ " heat_map.png"))



class Histogram_plotter:
    def __init__(self, codebooks_per_phylolevel, n_phylolevels, n_embed, 
                 converter, 
                 indx_to_label,
                 ckpt_path, directory):
        self.codebooks_per_phylolevel = codebooks_per_phylolevel
        self.n_phylolevels = n_phylolevels
        self.n_embed = n_embed
        self.converter = converter
        self.ckpt_path = ckpt_path
        self.directory = directory
        self.indx_to_label = indx_to_label
        
    def plot_histograms(self, histograms, species_indx, is_nonattribute=False, prefix="species"):
        fig, axs = plt.subplots(self.codebooks_per_phylolevel, self.n_phylolevels, figsize = (5*self.n_phylolevels,30))
        for i, ax in enumerate(axs.reshape(-1)):
            ax.hist(histograms[i], density=True, range=(0, self.n_embed-1), bins=self.n_embed)
            
            if not is_nonattribute:
                code_location, level = self.converter.get_code_reshaped_index(i)
                ax.set_title("code "+ str(code_location) + "/level " +str(level))
            else:
                ax.set_title("code "+ str(i))
        
        plt.show()
        sub_dir = 'attribute' if not is_nonattribute else 'non_attribute'
        fig.savefig(os.path.join(get_fig_pth(self.ckpt_path, postfix=self.directory+'/'+sub_dir), "{}_{}_{}_hostogram.png".format(prefix, species_indx, self.indx_to_label[species_indx])))
        plt.close(fig)