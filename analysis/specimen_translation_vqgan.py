
from scripts.loading_utils import load_config, load_model
from scripts.data.custom import CustomTest as CustomDataset
from scripts.models.vqgan import VQModel
from scripts.plotting_utils import get_fig_pth
from scripts.models.cwautoencoder import CWmodelVQGAN

import os
import numpy as np
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
    model_name = configs_yaml.model_name if "model_name" in configs_yaml.keys() else 'VQGAN'

    # Load model
    if model_name=='CW':
        model_type=CWmodelVQGAN
    else :
        model_type=VQModel
    
    # load image
    dataset = CustomDataset(size, file_list_path, add_labels=True)
    
    # Load model
    config = load_config(yaml_path, display=False)
    model = load_model(config, ckpt_path=ckpt_path, cuda=(DEVICE is not None), model_type=model_type)
    
    for indx__ in tqdm.tqdm(range(count)):
        image_index1 = image_index1_+indx__
        image_index2 = image_index2_+indx__
        
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
                continue
            
            code1_modified[i] = code2[i]
            
            dec_image = model.decode_code(code1_modified)
            
            
            if (key_image_helper.isKeyImage(i)):
                dec_images.insert(-2, dec_image)
                
                change = {
                    'code_index': i,
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
                    title ="translation " + str(int(change['code_match_%'])) + "%"
                    change_index = change_index+ 1
                else:
                    if img_index==1:
                        title="reconstruction"
                    else:
                        title="translation complete"
                
            axes.set_title(title, fontdict=None, loc='center', color = "k")
            axes.tick_params(left=False, bottom=False, labelbottom=False, labelleft=False)
                
            img_index = img_index + 1
                
        plt.show()
                
        fig_path = get_fig_pth(ckpt_path)
        filename = "from {} to {}.png".format(image_index1, image_index2)
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
        default="analysis/configs/specimen_translation_vqgan.yaml",
    )
    
    cfg, _ = parser.parse_known_args()
    configs = OmegaConf.load(cfg.config)
    cli = OmegaConf.from_cli()
    config = OmegaConf.merge(configs, cli)
    print(config)
    
    main(config)