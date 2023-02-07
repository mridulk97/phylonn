
# import copy
import os
import pickle
from analysis.replace_codes import get_entropy_ordering
import numpy as np
from taming.analysis_utils import Embedding_Code_converter
# import os
# from taming.data.phylogeny import Phylogeny
from taming.loading_utils import load_config, load_phylovqvae
from taming.data.custom import CustomTest as CustomDataset
# from taming.models.cond_transformer import Phylo_Net2NetTransformer

import taming.constants as CONSTANTS
from taming.models.phyloautoencoder import PhyloVQVAE
from taming.modules.losses.lpips import LPIPS
# from taming.modules.losses.phyloloss import parse_phyloDistances
from taming.plotting_utils import get_fig_pth


import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

import torch

from omegaconf import OmegaConf
import argparse



class KeyImageHelper:
    def __init__(self, cb_per_level, n_phylolevels, get_code_reshaped_index):
        self.cb_per_level = cb_per_level
        self.n_phylolevels = n_phylolevels
        self.get_code_reshaped_index = get_code_reshaped_index
        self.n_phylocodes = self.n_phylolevels*self.cb_per_level
        
        self.update = True
    
    
    
    def new_update(self, i):
        if i < self.n_phylocodes:
            code_index_relative, _ = self.get_code_reshaped_index(i)
            if code_index_relative == 0:
                self.update = True
        
        return self.update
        
    def isKeyImage(self, i):
        if i < self.n_phylocodes:
            if self.update:
                self.update = False
                return True
            
        return False
    
    
    
class KeyImageEntropyHelper:
    def __init__(self, cb_per_level, n_phylolevels, n_nonphylocodes, get_code_reshaped_index):
        self.cb_per_level = cb_per_level
        self.n_phylolevels = n_phylolevels
        self.n_nonphylocodes = n_nonphylocodes
        self.get_code_reshaped_index = get_code_reshaped_index
        
        self.update = True
        self.changed_codes = []
        for i in range(self.n_phylolevels):
            self.changed_codes.append([])
    
    
    
    def new_update(self, indx, i):
        nonphylo_codes_count = self.cb_per_level*self.n_nonphylocodes
        
        is_phylo_end = indx%nonphylo_codes_count
        if is_phylo_end == 0:
            self.update = True
            if indx > 0:
                self.changed_codes[0].append(i)
            # print('key nonphylo')
        elif indx>nonphylo_codes_count:
            _, phylo_level = self.get_code_reshaped_index(i)
            # print('phylolevel', phylo_level)
            self.changed_codes[phylo_level].append(i)
            # print(self.changed_codes)
            if len(self.changed_codes[phylo_level]) == self.cb_per_level:
                self.update = True
                # print('key phylo')
        # print('-----')
                
        return self.update
        
    def isKeyImage(self, i):
        _, lvl = self.get_code_reshaped_index(i)
        if self.update:
            self.update = False
            # print(lvl, 'is key img')
            return True
            
        
        return False
            
def populate_resnet_scores(resnet_models, dec_image, change):
    if resnet_models is not None:
        for indx, k in enumerate(resnet_models):
            change['lvl'+str(indx)+"_class"] = torch.argmax(k(dec_image), dim=1).item()
    return change 


def rearrange_phylo_codeorder(code, get_code_reshaped_index, n_nonphylo, n_phylolevel, n_code_per_level):
    # print(code)
    n_phylocodes = n_phylolevel*n_code_per_level
    new_code = code.copy()
    
    i = 0
    for l in range(n_phylolevel):
        for k in range(n_code_per_level):
            i_ = get_code_reshaped_index(k,l)
            new_code[i] = i_
            i = i+1
    
    # print(new_code)
    # raise
    return new_code

@torch.no_grad()
def main(configs_yaml):
    yaml_path = configs_yaml.yaml_path
    ckpt_path = configs_yaml.ckpt_path
    DEVICE= configs_yaml.DEVICE
    image_index1= configs_yaml.image_index1
    image_index2= configs_yaml.image_index2
    # phyloDistances_string = configs_yaml.phyloDistances_string
    # phylogeny_path = configs_yaml.phylogeny_path
    size= configs_yaml.size
    file_list_path= configs_yaml.file_list_path
    # plt_path = configs_yaml.plt_path
    order = configs_yaml.order
    show_only_key_imgs = configs_yaml.show_only_key_imgs
    according_to_target = configs_yaml.according_to_target
    
    bb_model_paths = configs_yaml.bb_model_paths
    resnet_models = None
    if bb_model_paths is not None:
        resnet_models = []
        for i in bb_model_paths:
            model_ = torch.load(i).eval()
            if DEVICE is not None:
                model_ = model_.to(DEVICE)
            resnet_models.append(model_)
        
    plt_path = order if order is not None else "default"
    
    # phylo_distances = parse_phyloDistances(phyloDistances_string)
    
    # load image
    dataset = CustomDataset(size, file_list_path, add_labels=True)
    
    # Load model
    config = load_config(yaml_path, display=False)
    model = load_phylovqvae(config, ckpt_path=ckpt_path, cuda=(DEVICE is not None), model_type=PhyloVQVAE)
    
    get_code_reshaped_index = model.phylo_disentangler.embedding_converter.get_code_reshaped_index
    n_phylocodes = model.phylo_disentangler.n_phylolevels*model.phylo_disentangler.codebooks_per_phylolevel
    n_nonphylocodes = model.phylo_disentangler.n_phylolevels*model.phylo_disentangler.n_levels_non_attribute
    
    # load image. get class and zq_phylo and z_qnonattr
    classes = []
    codes = []
    dec_images = []
    filenames = []
    converter_phylo = None
    converter_nonattr = None
    len_phylo = None
    # imgs = []
    for img_indx in [image_index1, image_index2]:
        specimen = dataset.data[img_indx]
        processed_img = torch.Tensor(specimen['image']).unsqueeze(0).to(DEVICE)
        processed_img = processed_img.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format)
        
        dec_image, _, _, in_out_disentangler = model(processed_img)
        dec_images.append(dec_image)
        if img_indx == image_index1:
            dec_images.insert(0, processed_img)
        else:
            dec_images.append(processed_img)
            
        q_phylo = in_out_disentangler[CONSTANTS.QUANTIZED_PHYLO_OUTPUT] 
        q_non_attr = in_out_disentangler[CONSTANTS.QUANTIZED_PHYLO_NONATTRIBUTE_OUTPUT] 

        if converter_phylo is None:
            converter_phylo = Embedding_Code_converter(model.phylo_disentangler.quantize.get_codebook_entry_index, model.phylo_disentangler.quantize.embedding, q_phylo[0, :, :, :].shape)
            converter_nonattr = Embedding_Code_converter(model.phylo_disentangler.quantize.get_codebook_entry_index, model.phylo_disentangler.quantize.embedding, q_non_attr[0, :, :, :].shape)
        
        all_code_indices_phylo = converter_phylo.get_phylo_codes(q_phylo[0, :, :, :].unsqueeze(0))
        all_code_indices_nonattr = converter_nonattr.get_phylo_codes(q_non_attr[0, :, :, :].unsqueeze(0))
        if len_phylo is None:
            len_phylo = all_code_indices_phylo.view(1, -1).shape[-1]
        
        all_code = torch.cat([all_code_indices_phylo, all_code_indices_nonattr], dim=1)
        codes.append(all_code)
        
        species_index = specimen['class']
        filenames.append(os.path.basename(specimen['file_path_']))
        classes.append(species_index)
        print(specimen['file_path_'], specimen['class'])
        
        
    changes = []
    code1_modified =torch.clone(codes[0])
    code1 = codes[0]
    code2 = codes[1]
    
    
    if order is None or order == "reverse":
        order_ = range(codes[0].shape[1])
        order_ = rearrange_phylo_codeorder(list(order_), get_code_reshaped_index, n_nonphylocodes, model.phylo_disentangler.n_phylolevels, model.phylo_disentangler.codebooks_per_phylolevel)
        if order == "reverse":
            order_ = list(reversed(order_))
    elif order == "entropy":
        # Get code and location ordering
        histograms_file = os.path.join(get_fig_pth(ckpt_path, postfix=CONSTANTS.HISTOGRAMS_FOLDER), CONSTANTS.HISTOGRAMS_FILE)
        histogram_file_exists = os.path.exists(histograms_file)
        if not histogram_file_exists:
            raise "histograms have not been generated. Run code_histogram.py first! Defaulting to index ordering"
        hist_arr, hist_arr_nonattr = pickle.load(open(histograms_file, "rb"))
        
        # by entropy over all
        # hist_total = []
        # for indx, i in enumerate(hist_arr):
        #     hist_total.append(hist_arr[indx] + hist_arr_nonattr[indx])
        # order_ = np.flip(get_entropy_ordering(hist_total[classes[-1]]))# gives from least important to most important according to target species.
        
        # by entropy over nonphylo then phylo
        # target_class = classes[-1 if according_to_target else 0]
        # entropy_phylo = np.flip(get_entropy_ordering(hist_arr[target_class]))
        # entropy_nonphylo = np.flip(get_entropy_ordering(hist_arr_nonattr[target_class]) + entropy_phylo.shape[0])
        # order_ = np.concatenate((entropy_nonphylo, entropy_phylo))
        
        # by entrop over levels
        target_class = classes[-1 if according_to_target else 0]
        entropy_nonphylo = np.flip(get_entropy_ordering(hist_arr_nonattr[target_class]) + n_phylocodes)
        hist_phylo = hist_arr[target_class]
        hist_levels = []
        for i in range(n_phylocodes):
            hist_levels.append([])
        for i in range(len(hist_phylo)):
            relative_indx, lvl = get_code_reshaped_index(i)
            # print(i, lvl, relative_indx)
            hist_levels[lvl].append(hist_phylo[i])
        # for lvl in range(model.phylo_disentangler.n_phylolevels):
        #     print('hist_levels', lvl, hist_levels[lvl])
        for lvl in range(model.phylo_disentangler.n_phylolevels):
            hist_levels[lvl] = np.flip(get_entropy_ordering(hist_levels[lvl])) #+ relative_index*model.phylo_disentangler.n_phylolevels
            # print('hist_levels____', hist_levels[lvl])    
        # for lvl in range(model.phylo_disentangler.n_phylolevels):
        #     for relative_index in range(model.phylo_disentangler.codebooks_per_phylolevel):
        #         hist_levels[lvl][relative_index] = hist_levels[lvl][relative_index] + lvl*model.phylo_disentangler.codebooks_per_phylolevel
        #     print('hist_levels____2', hist_levels[lvl])   
        entropy_phylo = np.zeros(len(hist_phylo), dtype=int)
        o = 0
        for lvl in range(model.phylo_disentangler.n_phylolevels):
            for relative_index in range(model.phylo_disentangler.codebooks_per_phylolevel):
                # abs_indx = get_code_reshaped_index(relative_index, lvl)
                abs_indx = get_code_reshaped_index(hist_levels[lvl][relative_index], lvl)
                # print(abs_indx, lvl, relative_indx, relative_index*model.phylo_disentangler.n_phylolevels)
                # entropy_phylo[abs_indx] = hist_levels[lvl][relative_index]
                entropy_phylo[o] = abs_indx
                o = o+1
        # print('entropy_phylo', entropy_phylo) 
        order_ = np.concatenate((entropy_nonphylo, entropy_phylo))
        # print(order_)
        # raise
        
        
        # print(order, type(order))
        # raise
    else:
        raise order + " order is not valid"
    if show_only_key_imgs:
        if order != "entropy":
            key_image_helper = KeyImageHelper(model.phylo_disentangler.codebooks_per_phylolevel, model.phylo_disentangler.n_phylolevels, get_code_reshaped_index)
        else:
            key_image_helper = KeyImageEntropyHelper(model.phylo_disentangler.codebooks_per_phylolevel, model.phylo_disentangler.n_phylolevels, model.phylo_disentangler.n_levels_non_attribute, get_code_reshaped_index)
        

    perceptual_loss = LPIPS().eval()
    if DEVICE is not None:
        perceptual_loss = perceptual_loss.cuda()
        
    for indx_, i in enumerate(order_):    
        if show_only_key_imgs:
            if order != "entropy":
                key_image_helper.new_update(i)
            else:
                key_image_helper.new_update(indx_, i)
            
        if code1[0, i] == code2[0, i]:
            continue #NOTE: This means a code might be skipped from key images of it is identical. Does not seem to be a bug but it is disorienting 
        
        code1_modified[0, i] = code2[0, i]
        
        # print(code1_modified.shape, len_phylo)
        z_phylo = converter_phylo.get_phylo_embeddings(code1_modified[:, :len_phylo])
        z_nonphylo = converter_nonattr.get_phylo_embeddings(code1_modified[:, len_phylo:])
        dec_image, _, _, in_out_disentangler = model.decode(z_phylo, z_nonphylo)
        
        
        # print(indx_, i, get_code_reshaped_index(i)[1] if i < n_phylocodes else 'nonphylo')
        if (not show_only_key_imgs) or (key_image_helper.isKeyImage(i)):
            dec_images.insert(-2, dec_image)
            
            change = {
                'indx_': indx_,
                'perceptual_difference': perceptual_loss(dec_image, dec_images[1]).item(),
                'code_index': i,
                'phylo_level': get_code_reshaped_index(i)[1] if i < n_phylocodes else 'nonphylo',  #int(i/model.phylo_disentangler.codebooks_per_phylolevel),
                'from_code': code1[0, i].detach().item(),
                'to_code': code2[0, i].detach().item(),
                'code_match_%': 100-torch.cdist(code1_modified.float(), code2.float(), p=0).item()*100/code2.shape[1]
            }
            
            changes.append(change)
            # print(change)
            # print('-----------')
        
    fig = plt.figure(1, (3.5*len(dec_images), 5.))
    grid = ImageGrid(fig, 111,
                    nrows_ncols=(1, len(dec_images)),
                    axes_pad=0.4,
                    )
    
    img_index = 0
    change_index = 0
    # print('len(dec_images)', len(dec_images))
    # print('changes',len(changes))
    for axes in grid:  
        img_ = dec_images[img_index]
        img = ((img_.squeeze().T.detach().cpu().numpy()+1.)*127.5).astype(np.uint8)
        axes.imshow(img)

        if img_index > 0 and img_index < len(dec_images)-1:
            if img_index > 1 and img_index < len(dec_images)-2:
                change = changes[change_index]
                title = "change# {} \n indx/lvl: {}/{} \n from {} to {} \n code match % {} \n diff {:.2f}".format(
                    change['indx_'], change['code_index'], change['phylo_level'], change['from_code'], change['to_code'], change['code_match_%'], change['perceptual_difference']
                )
                change_index = change_index+ 1
            else:
                title = "name: {} \n class {}".format(
                    filenames[0 if img_index==1 else 1], classes[0 if img_index==1 else 1]
                )
                
            if resnet_models is not None:
                populate_resnet_scores(resnet_models, img_, change) 
                title = title + "\n classifications: "
                for k in range(len(resnet_models)):
                    title = title + '{}/'.format(change['lvl'+str(k)+"_class"])
            
            axes.set_title(title, fontdict=None, loc='center', color = "k")
            
        img_index = img_index + 1
            
    plt.show()
            
    fig_path = get_fig_pth(ckpt_path)
    filename = "{} from {} to {}.png".format(plt_path, image_index1, image_index2)
    fig_path = str(fig_path)+ "/" + filename
    fig.savefig(fig_path)
                
                
                
            
            
        
    
        
            
        
            
        
    
        
        
    
        
    
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-n",
        "--config",
        type=str,
        nargs="?",
        const=True,
        default="analysis/configs/specimen_diff.yaml",
    )
    
    cfg, _ = parser.parse_known_args()
    # cfg = parser.config
    configs = OmegaConf.load(cfg.config)
    cli = OmegaConf.from_cli()
    config = OmegaConf.merge(configs, cli)
    print(config)
    
    main(config)