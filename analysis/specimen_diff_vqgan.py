
# import copy
import os
import numpy as np
from taming.loading_utils import load_config, load_phylovqvae
from taming.data.custom import CustomTest as CustomDataset

from taming.models.vqgan import VQModel
from taming.plotting_utils import get_fig_pth


import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

import torch

from omegaconf import OmegaConf
import argparse
import tqdm
from pathlib import Path



class KeyImageHelper:
    def __init__(self, total_num_codes, keyimagerate):
        self.total_num_codes = total_num_codes
        self.keyimagerate = keyimagerate
        
        self.update = True
    
    
    
    def new_update(self, i):
        if i%self.keyimagerate == 0:
            self.update = True
        
        return self.update
        
    def isKeyImage(self, i):
        if self.update:
            self.update = False
            return True
            
        return False
    
    
            
def populate_resnet_scores(resnet_model, dec_image, change):
    if resnet_model is not None:
        change["class"] = torch.argmax(resnet_model(dec_image), dim=1).item()
    return change 



@torch.no_grad()
def main(configs_yaml):
    yaml_path = configs_yaml.yaml_path
    ckpt_path = configs_yaml.ckpt_path
    DEVICE= configs_yaml.DEVICE
    image_index1_= configs_yaml.image_index1
    image_index2_= configs_yaml.image_index2
    count= configs_yaml.count
    size= configs_yaml.size
    file_list_path= configs_yaml.file_list_path
    keyimgrate= configs_yaml.keyimgrate
    
    bb_model_path = configs_yaml.bb_model_path
        
    plt_path = "default"
    
    # load image
    dataset = CustomDataset(size, file_list_path, add_labels=True)
    
    # Load model
    config = load_config(yaml_path, display=False)
    model = load_phylovqvae(config, ckpt_path=ckpt_path, cuda=(DEVICE is not None), model_type=VQModel)
    
    for indx__ in tqdm.tqdm(range(count)):
        image_index1 = image_index1_+indx__
        image_index2 = image_index2_+indx__
        
        # load image. get class and zq_phylo and z_qnonattr
        classes = []
        codes = []
        dec_images = []
        filenames = []

        for img_indx in [image_index1, image_index2]:
            specimen = dataset.data[img_indx]
            processed_img = torch.Tensor(specimen['image']).unsqueeze(0).to(DEVICE)
            processed_img = processed_img.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format)
            
            quant, diff, info = model.encode(processed_img)
            dec_image = model.decode(quant)
        
            dec_images.append(dec_image)
            if img_indx == image_index1:
                dec_images.insert(0, processed_img)
            else:
                dec_images.append(processed_img)
                
            all_code = info[2]
            codes.append(all_code)
            
            species_index = specimen['class']
            filenames.append(os.path.basename(specimen['file_path_']))
            classes.append(species_index)
            print(specimen['file_path_'], specimen['class'])
            
            
        changes = []
        code1_modified =torch.clone(codes[0])
        code1 = codes[0]
        code2 = codes[1]
        
        order_ = range(codes[0].shape[0])
       
        key_image_helper = KeyImageHelper(model.quantize.n_e, keyimgrate)
            
        for indx_, i in enumerate(order_):    
            key_image_helper.new_update(i)
                
            if (not key_image_helper.update) and code1[i] == code2[i]:
                continue #NOTE: This means a code might be skipped from key images of it is identical. Does not seem to be a bug but it is disorienting 
            
            code1_modified[i] = code2[i]
            
            dec_image = model.decode_code(code1_modified)
            
            
            if (key_image_helper.isKeyImage(i)):
                dec_images.insert(-2, dec_image)
                
                change = {
                    # 'indx_': indx_,
                    'code_index': i,
                    # 'phylo_level': get_code_reshaped_index(i)[1] if i < n_phylocodes else 'nonphylo',  #int(i/model.phylo_disentangler.codebooks_per_phylolevel),
                    'from_code': code1[i].detach().item(),
                    'to_code': code2[i].detach().item(),
                    'code_match_%': 100-torch.cdist(code1_modified.view((1, -1)).float(), code2.view((1, -1)).float(), p=0).item()*100/code2.shape[0]
                }
                
                changes.append(change)
            
        fig = plt.figure(1, (2.5*len(dec_images), 4.))
        grid = ImageGrid(fig, 111,
                        nrows_ncols=(1, len(dec_images)),
                        )
        
        img_index = 0
        change_index = 0
        for axes in grid:  
            img_ = dec_images[img_index]
            img = ((np.fliplr(img_.squeeze().T.detach().cpu().numpy())+1.)*127.5).astype(np.uint8)
            axes.imshow(img, aspect='auto')
            if img_index==0:
                title = "source image"
            elif img_index == len(dec_images)-1:
                title = "target image"

            if img_index > 0 and img_index < len(dec_images)-1:
                if img_index > 1 and img_index < len(dec_images)-2:
                    change = changes[change_index]
                    # title = "indx {} \n from {} to {} \n code match % {}".format(
                    #     change['code_index'], change['from_code'], change['to_code'], change['code_match_%']
                    # )
                    title ="translation " + str(int(change['code_match_%'])) + "%"
                    change_index = change_index+ 1
                else:
                    # title = "name: {} \n class {}".format(
                    #     filenames[0 if img_index==1 else 1], classes[0 if img_index==1 else 1]
                    # )
                    if img_index==1:
                        title="reconstruction"
                    else:
                        title="translation complete" #TODO: hardcoded!
                    
                if bb_model_path is not None:
                    populate_resnet_scores(bb_model_path, img_, change) 
                    title = title + "\n classifications: "
                    title = title + '{}/'.format(change["class"])
                
            axes.set_title(title, fontdict=None, loc='center', color = "k")
            axes.tick_params(left=False, bottom=False, labelbottom=False, labelleft=False)
                
            img_index = img_index + 1
                
        plt.show()
                
        fig_path = get_fig_pth(ckpt_path)
        filename = "{} from {} to {}.png".format(plt_path, image_index1, image_index2)
        fig_path = os.path.join(str(fig_path), "transitions", dataset.indx_to_label[classes[0]] + ' to ' + dataset.indx_to_label[classes[1]])
        Path(fig_path).mkdir(parents=True, exist_ok=True)
        fig.savefig(os.path.join(fig_path, filename),bbox_inches='tight',dpi=300)
        
        plt.close()
                    
                    
                    
                
                
            
        
            
                
            
                
        
    
        
        
    
        
    
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-n",
        "--config",
        type=str,
        nargs="?",
        const=True,
        default="analysis/configs/specimen_diff_vqgan.yaml",
    )
    
    cfg, _ = parser.parse_known_args()
    # cfg = parser.config
    configs = OmegaConf.load(cfg.config)
    cli = OmegaConf.from_cli()
    config = OmegaConf.merge(configs, cli)
    print(config)
    
    main(config)