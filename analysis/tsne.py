from scripts.analysis_utils import get_phylomapper_from_config
from scripts.data.phylogeny import Phylogeny
from scripts.loading_utils import load_config, load_phylovqvae
from scripts.models.phyloautoencoder import PhyloVQVAE
from scripts.models.vqgan import VQModel
from scripts.plotting_utils import get_fig_pth
from scripts.data.utils import custom_collate
from scripts.data.custom import CustomTest as CustomDataset
import scripts.constants as CONSTANTS

import argparse
import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from torchvision import transforms
from tqdm import tqdm
import torch
from PIL import Image, ImageDraw, ImageFont
import matplotlib
from matplotlib.pyplot import imshow, show, plt
import os
import pandas as pd
import seaborn as sns
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from sklearn.neighbors import NearestNeighbors


MAX_DIMS_PCA=100

def get_output(model, image):
    if type(model) == PhyloVQVAE:
        _, _, _, in_out_disentangler = model(image)
        return in_out_disentangler[CONSTANTS.QUANTIZED_PHYLO_OUTPUT]
    elif type(model) == VQModel: 
        quant, _, _ = model.encode(image)
        return quant
    else:
        raise "Model type unknown"
    
    



##############

# Given a dataloader, a model, and an activation layer, it displays an images tsne
def get_tsne(dataloader, model, path, 
    legend_labels=[CONSTANTS.DISENTANGLER_CLASS_OUTPUT], 
    cuda=None,
    which_tsne_plots = ['standard',  'knn']
    , file_prefix='default_name',
    phylomapper = None,
    phylogeny_knn=None):

    # Go thtough batches
    X = None
    for i, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
        image2 = batch['image']
        if cuda is not None:
            image2 = image2.cuda()
        image2 = image2.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format)
            
        features2 = get_output(model, image2)
        features2 = features2.detach().cpu().reshape(features2.shape[0], -1)
        X = features2 if X is None else torch.cat([X, features2]).detach()

    #PCA
    if X.shape[1]>MAX_DIMS_PCA:
        pca = PCA(n_components=MAX_DIMS_PCA)
        X = pca.fit_transform(X)

    # TNSE
    tsne = TSNE(n_components=2, learning_rate=150, verbose=2).fit_transform(X)
    tx, ty = tsne[:,0], tsne[:,1]
    tx = (tx-np.min(tx)) / (np.max(tx) - np.min(tx))
    ty = (ty-np.min(ty)) / (np.max(ty) - np.min(ty))

    if 'standard' in which_tsne_plots:
        plot_tsne_dots(dataloader, tx, ty, path, file_prefix, legend_labels, phylomapper)
    
    if ('knn' in which_tsne_plots):
        if phylogeny_knn is None:
            print('KNN plot cannot be calculated if phylogeny_knn is not passed! skipping...')
        if phylomapper is not None:
            print('KNN can only be calculated at level 3! skipping...')
        else:
            phylogeny_knn = Phylogeny(phylogeny_knn)
            plot_phylo_KNN(dataloader, tx, ty, path, file_prefix, phylogeny_knn)
            

    

    show(block=False)
    print('--------')
    








def avg_distances(fine_label, indexes, phylogeny, dataset):
    dist = 0
    label_list = phylogeny.getLabelList()
    for i in indexes:
        lbl = dataset[i]['class']
        dist = dist + phylogeny.get_distance(label_list[fine_label], label_list[lbl])
    result = dist/len(indexes)
    return result



def plot_phylo_KNN(dataloader, tx, ty, path, file_prefix, phylogeny_knn, n_neighbors=5):
    # Parse through the dataloader images
    KNN_values = None

    coord = np.hstack((np.array(tx).reshape(-1,1),np.array(ty).reshape(-1,1)))
    nbrs = NearestNeighbors(n_neighbors=n_neighbors).fit(coord)
    _, indexes = nbrs.kneighbors(coord)
    
    acc=0
    for i, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
        fine_label = batch['class']
        
        KNN_values_ = torch.tensor([avg_distances(x, indexes[acc+i_], phylogeny_knn, dataloader.dataset) for i_, x in  enumerate(fine_label)]) 
        KNN_values = KNN_values_ if KNN_values is None else torch.cat([KNN_values, KNN_values_]).detach()
        
        acc = acc + len(fine_label)


    df = pd.DataFrame()
    df['tsne-x'] = tx
    df['tsne-y'] = ty
    df['KNN_phylo_dist'] = KNN_values

    matplotlib.pyplot.figure(figsize=(16,10))
    sns_plot = sns.scatterplot(
        x="tsne-x", y="tsne-y",
        hue='KNN_phylo_dist',
        palette="Oranges",
        data=df,
        legend=False
    )

    norm = plt.Normalize(df['KNN_phylo_dist'].min(), df['KNN_phylo_dist'].max())
    sm = plt.cm.ScalarMappable(cmap="Oranges", norm=norm)
    sm.set_array([])

    # Remove the legend and add a colorbar
    sns_plot.figure.colorbar(sm)

    fig = sns_plot.get_figure()
    save_path = os.path.join(path, file_prefix+"_tsne_KNN_phylo.png")
    print(save_path)
    fig.savefig(save_path,bbox_inches='tight',dpi=300)  


def plot_tsne_dots(dataloader, tx, ty, path, file_prefix, legend_labels=[CONSTANTS.DISENTANGLER_CLASS_OUTPUT], phylomapper=None):
    labels = {}
    file_names=[]
    for i, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
        file_name = batch['file_path_']
        for j in legend_labels:
            labels[j] = batch[j] if j not in labels else torch.cat([labels[j], batch[j]]).detach()
        file_names = file_names + file_name


    df = pd.DataFrame()
    df['tsne-x'] = tx
    df['tsne-y'] = ty
    for j in legend_labels:
        if phylomapper is not None:
            labels[j] = phylomapper.get_mapped_truth(labels[j])
            
        labels[j] = labels[j].tolist()
        df[j] = labels[j]
        
    
    if phylomapper is not None:
        file_prefix = file_prefix+"_level"+str(phylomapper.level)
    
    for j in legend_labels:    
        matplotlib.pyplot.figure(figsize=(16,10))
        sns_plot = sns.scatterplot(
            x="tsne-x", y="tsne-y",
            hue=j,
            palette=sns.color_palette("hls", len(set(labels[j]))),
            data=df,
            legend=False
        )
        fig = sns_plot.get_figure()
        fig.savefig(os.path.join(path, file_prefix+"_legend_" + j +"_tsne_dots.png"),bbox_inches='tight',dpi=300)






def main(configs_yaml):
    yaml_path = configs_yaml.yaml_path
    ckpt_path = configs_yaml.ckpt_path
    dataset_path = configs_yaml.dataset_path 
    file_prefix = configs_yaml.file_prefix
    img_res = configs_yaml.img_res
    which_tsne_plots = configs_yaml.which_tsne_plots
    legend_labels = configs_yaml.legend_labels
    DEVICE= configs_yaml.DEVICE
    batch_size = configs_yaml.batch_size
    num_workers = configs_yaml.num_workers
    phylogeny_knn = configs_yaml.phylogeny_knn
    phylogeny_path = configs_yaml.phylogeny_path
    
    isOriginalVQGAN = configs_yaml.isOriginalVQGAN if "isOriginalVQGAN" in configs_yaml.keys() else False
    
    # get phylogeny
    phylomapper=None
    if phylogeny_path is not None:
        level = configs_yaml.level
        phyloDistances_string = configs_yaml.phyloDistances_string
        phylomapper = get_phylomapper_from_config(Phylogeny(phylogeny_path), phyloDistances_string, level)
    
    # get dataset
    dataset = CustomDataset(img_res, dataset_path, add_labels=True)
    dataloader = DataLoader(dataset.data, batch_size=batch_size, num_workers=num_workers, collate_fn=custom_collate)
    
    # Load model
    if isOriginalVQGAN:
        model_type=VQModel
    else:
        model_type=PhyloVQVAE
        
    config = load_config(yaml_path, display=False)
    model = load_phylovqvae(config, ckpt_path=ckpt_path, cuda=(DEVICE is not None), model_type=model_type)
    
    with torch.no_grad():
        get_tsne(dataloader, model, get_fig_pth(ckpt_path, postfix=CONSTANTS.TSNE_FOLDER), 
            legend_labels=legend_labels,
            which_tsne_plots = which_tsne_plots
            , file_prefix=file_prefix,
            phylomapper=phylomapper,
            phylogeny_knn=phylogeny_knn,
            cuda=DEVICE)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-n",
        "--config",
        type=str,
        nargs="?",
        const=True,
        default="analysis/configs/tsne.yaml",
    )
    
    cfg, _ = parser.parse_known_args()
    configs = OmegaConf.load(cfg.config)
    cli = OmegaConf.from_cli()
    config = OmegaConf.merge(configs, cli)
    print(config)
    
    main(config)
