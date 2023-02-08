import argparse
import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from taming.analysis_utils import get_phylomapper_from_config
from taming.data.phylogeny import Phylogeny
from taming.loading_utils import load_config, load_phylovqvae, load_CWVQGAN
from taming.models.phyloautoencoder import PhyloVQVAE
from taming.models.cwautoencoder import CWmodelVQGAN
from taming.models.vqgan import VQModel
from taming.plotting_utils import get_fig_pth
from torchvision import transforms
from tqdm import tqdm
import torch
from PIL import Image, ImageDraw, ImageFont
import matplotlib
from matplotlib.pyplot import imshow, show
import os
import pandas as pd
import seaborn as sns
import mpld3
import matplotlib.pyplot as plt
from omegaconf import OmegaConf
from taming.data.utils import custom_collate
from taming.data.custom import CustomTest as CustomDataset
from torch.utils.data import DataLoader
from sklearn.neighbors import NearestNeighbors
import taming.constants as CONSTANTS

MAX_DIMS_PCA=100

def get_output(model, image):
    if type(model) == PhyloVQVAE:
        _, _, _, in_out_disentangler = model(image)
        return in_out_disentangler[CONSTANTS.QUANTIZED_PHYLO_OUTPUT]
    elif type(model) == VQModel: 
        quant, _, _ = model.encode(image)
        return quant
    elif type(model) == CWmodelVQGAN:
        # quant, _, _ = model.encode(image)
        h = model.encoder(image)
        h = model.quant_conv(h)
        return h
    else:
        raise "Model type unknown"
    
    


##############

# Given a dataloader, a model, and an activation layer, it displays an images tsne
def get_tsne(dataloader, model, path, 
    legend_labels=[CONSTANTS.DISENTANGLER_CLASS_OUTPUT], 
    cuda=None,
    which_tsne_plots = ['standard', 'images', 'incorrect']
    , file_prefix='default_name'
    , img_res=None,
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



    if img_res is not None and 'images' in which_tsne_plots:
        visualize_tsne_images(dataloader, tx, ty, img_res, path, file_prefix, phylomapper, cuda)

    if 'standard' in which_tsne_plots:
        plot_tsne_dots(dataloader, tx, ty, path, file_prefix, legend_labels, phylomapper)
    
    if 'incorrect' in which_tsne_plots:
        plot_correct_incorrect(dataloader, tx, ty, path, file_prefix, model, phylomapper, cuda)
    
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
        legend=False #legend="full"
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


def plot_correct_incorrect(dataloader, tx, ty, path, file_prefix, model=None, phylomapper=None, cuda=None):
    # Parse through the dataloader images
    equality = None

    for i, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
        fine_label = batch['class']
        if phylomapper is not None:
            fine_label = phylomapper.get_mapped_truth(fine_label)
            
        image2 = batch['image']
        
        if cuda is not None:
            image2 = image2.cuda()
            fine_label = fine_label.cuda()
        
        _, _, _, in_out_disentangler = model(image2)
        pred_ = in_out_disentangler[CONSTANTS.DISENTANGLER_CLASS_OUTPUT]
        _, pred_ = torch.max(torch.nn.Softmax(dim=1)(pred_),1)
        equality_ = torch.eq(pred_, fine_label)
        equality = equality_ if equality is None else torch.cat([equality, equality_]).detach()

    df = pd.DataFrame()
    df['tsne-x'] = tx
    df['tsne-y'] = ty
    fine_labels = equality.tolist()
    df['isCorrect'] = fine_labels

    matplotlib.pyplot.figure(figsize=(16,10))
    sns_plot = sns.scatterplot(
        x="tsne-x", y="tsne-y",
        hue='isCorrect',
        palette=sns.color_palette("hls", len(set(fine_labels))),
        data=df,
        legend=False #"full"
    )
    fig = sns_plot.get_figure()
    fig.savefig(os.path.join(path, file_prefix+"_tsne_correctPrediction.png"),bbox_inches='tight',dpi=300)


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
            legend=False# legend="full"
        )
        # tooltip_label = [str(j) for i in range(len(labels))] #TODO: is this i a j?
        # tooltip = mpld3.plugins.PointLabelTooltip(sns_plot, labels=tooltip_label)
        fig = sns_plot.get_figure()
        # mpld3.plugins.connect(fig, tooltip)
        fig.savefig(os.path.join(path, file_prefix+"_legend_" + j +"_tsne_dots.png"),bbox_inches='tight',dpi=300)




def visualize_tsne_images(dataloader, tx, ty, img_res, path, file_prefix, phylomapper, cuda=None):    
    images = None
    fine_labels=None
    file_names=[]
    for i, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
        image2 = batch['image']
        fine_label = batch['class']
        file_name = batch['file_path_']
        
            
        if phylomapper is not None:
            fine_label = phylomapper.get_mapped_truth(fine_label)
        
        if cuda is not None:
            image2 = image2.cuda()
        image2 = image2.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format)
        image2 = ((image2 + 1.0)*127.5).clip(0,255).type(torch.uint8)
        images = image2 if images is None else torch.cat([images, image2]).detach()
        fine_labels = fine_label if fine_labels is None else torch.cat([fine_labels, fine_label]).detach()
        file_names = file_names + file_name

    # Construct the images
    width = 12000
    height = 9000
    max_dim = 150
    font = ImageFont.truetype(font='DejaVuSans.ttf', size=int(float(img_res) / 15))
    full_image = Image.new('RGB', (width, height))
    
    for img, x, y, file_name, fine_label in tqdm(zip(images, tx, ty, file_names, fine_labels), total=tx.shape[0]):
        tile = transforms.ToPILImage()(img).convert("RGB")
        # print(img.shape)
        # print(tile.size, tile.mode)
        
        draw = ImageDraw.Draw(tile)
        # draw.text((0, int(img_res*0.8)), "true: " + str(fine_label.item())+"\npredicted: "+str(pred.item()), (255,0,0), font=font)
        draw.text((0, int(img_res*0.8)), str(fine_label.item()), (255,0,0), font=font)
        
        rs = max(1, tile.width/max_dim, tile.height/max_dim)
        # print(rs, tile.width, tile.height, max_dim)
        tile = tile.resize((int(tile.width/rs), int(tile.height/rs)), Image.Resampling.LANCZOS)
        # print(tile.size, tile.mode)
        full_image.paste(tile, (int((width-max_dim)*x), height - int((height-max_dim)*y)))#, mask=tile.convert('RGBA'))
        # raise

    if phylomapper is not None:
        file_prefix = file_prefix+"_level"+str(phylomapper.level)
    
    matplotlib.pyplot.figure(figsize = (16,12))
    # full_image = full_image.transpose(Image.FLIP_TOP_BOTTOM)
    imshow(full_image)
    full_image.save(os.path.join(path, file_prefix+"_tsne_images.png"))
    




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
    
    isOriginalVQGAN = configs_yaml.isOriginalVQGAN if "isOriginalVQGAN" in configs_yaml.keys() else 'PhyloNN'
    
    phylogeny_path = configs_yaml.phylogeny_path
    print(type(phylogeny_path))
    
    phylomapper=None
    if phylogeny_path is not None:
        level = configs_yaml.level
        phyloDistances_string = configs_yaml.phyloDistances_string
    
        phylomapper = get_phylomapper_from_config(Phylogeny(phylogeny_path), phyloDistances_string, level)
    
    dataset = CustomDataset(img_res, dataset_path, add_labels=True)
    dataloader = DataLoader(dataset.data, batch_size=batch_size, num_workers=num_workers, collate_fn=custom_collate)
    
    # Load model
    if isOriginalVQGAN=='VQGAN':
        model_type=VQModel
    elif isOriginalVQGAN=='CW':
        model_type=CWmodelVQGAN
    else:
        model_type=PhyloVQVAE
        
    config = load_config(yaml_path, display=False)
    model = load_phylovqvae(config, ckpt_path=ckpt_path, cuda=(DEVICE is not None), model_type=model_type)
    # model.set_test_chkpt_path(ckpt_path)
    
    with torch.no_grad():
        get_tsne(dataloader, model, get_fig_pth(ckpt_path, postfix=CONSTANTS.TSNE_FOLDER), 
            legend_labels=legend_labels, #['fine'], 
            which_tsne_plots = which_tsne_plots #['standard', 'images', 'incorrect']
            , file_prefix=file_prefix
            , img_res=img_res,
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
    # cfg = parser.config
    configs = OmegaConf.load(cfg.config)
    cli = OmegaConf.from_cli()
    config = OmegaConf.merge(configs, cli)
    print(config)
    
    main(config)
