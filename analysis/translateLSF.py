import os
from scripts.loading_utils import load_config, load_model
from scripts.data.custom import CustomTest as CustomDataset
from scripts.models.LSFautoencoder import LSFVQVAE
from scripts.plotting_utils import get_fig_pth


import torch
from omegaconf import OmegaConf
import argparse
import tqdm
import torchvision.utils as vutils


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


def pic_morphing(model, output_dir, image_index1_, image_index2_, test_data, step=5):

    base_img = test_data.data[image_index1_]['image']
    target_img = test_data.data[image_index2_]['image']   

    if not os.path.isdir(f'{output_dir}'):
        os.makedirs(f'{output_dir}')

    imgs = []
    with torch.no_grad():
        data1 = base_img
        data1 = torch.tensor(data1).unsqueeze(0).permute(0, 3, 1, 2).to("cuda:0")
        imgs.append(data1.add(1.0).div(2.0).squeeze().rot90(3, [1, 2]))
        data2 = target_img
        data2 = torch.tensor(data2).unsqueeze(0).permute(0, 3, 1, 2).to("cuda:0")
        z1,_, _ = model.image2encoding(data1)
        z2,_, _ = model.image2encoding(data2)
        for s in range(0, step):
            z = (z1*(step-1-s)+z2*s)/(step-1)
            prod = model.encoding2image(z)
            prod.add_(1.0).div_(2.0)
            imgs.append(prod.squeeze().rot90(3, [1, 2]))

    imgs.append(data2.add(1.0).div(2.0).squeeze().rot90(3, [1, 2]))
    img = torch.stack(imgs)
    vutils.save_image(img, f'{output_dir}/Morphing_Pics_{image_index1_}_{image_index2_}.jpg', nrow=step+2, padding=4)

    return imgs


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
    
    # load image
    dataset = CustomDataset(size, file_list_path, add_labels=True)
    
    # Load model
    config = load_config(yaml_path, display=False)
    model = load_model(config, ckpt_path=ckpt_path, cuda=(DEVICE is not None), model_type=LSFVQVAE)

    for indx__ in tqdm.tqdm(range(count)):
        image_index1 = image_index1_+indx__
        image_index2 = image_index2_+indx__
        classes = []

        for img_indx in [image_index1, image_index2]:
            specimen = dataset.data[img_indx]
            
            species_index = specimen['class']
            classes.append(species_index)
    
        fig_path = get_fig_pth(ckpt_path)
        fig_path = os.path.join(str(fig_path), "translations", dataset.indx_to_label[classes[0]] + ' to ' + dataset.indx_to_label[classes[1]])
        
        pic_morphing(model, fig_path, image_index1, image_index2, dataset, step=6)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-n",
        "--config",
        type=str,
        nargs="?",
        const=True,
        default="analysis/configs/translateLSF.yaml",
    )
    
    cfg, _ = parser.parse_known_args()
    configs = OmegaConf.load(cfg.config)
    cli = OmegaConf.from_cli()
    config = OmegaConf.merge(configs, cli)
    print(config)
    
    main(config)