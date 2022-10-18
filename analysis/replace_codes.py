

from taming.loading_utils import load_config, load_phylovqvae
from taming.plotting_utils import save_image_grid
from taming.data.custom import CustomTest as CustomDataset
from taming.data.utils import custom_collate
from taming.analysis_utils import Embedding_Code_converter

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import taming.constants as CONSTANTS





###################

# parameters
# best 256 with phylo   
# yaml_path = "/home/elhamod/projects/taming-transformers/logs/2022-10-07T12-09-20_Phylo-VQVAE256img-afterhyperp/configs/2022-10-07T12-09-20-project.yaml"
# ckpt_path = "/home/elhamod/projects/taming-transformers/logs/2022-10-07T12-09-20_Phylo-VQVAE256img-afterhyperp/checkpoints/last.ckpt"
# 256 with korthogonality
# yaml_path = "/home/elhamod/projects/taming-transformers/logs/2022-10-13T01-13-29_Phylo-VQVAE256img-afterhyperp-kernelorthogonality/configs/2022-10-13T01-13-29-project.yaml"
# ckpt_path = "/home/elhamod/projects/taming-transformers/logs/2022-10-13T01-13-29_Phylo-VQVAE256img-afterhyperp-kernelorthogonality/checkpoints/last.ckpt"
# 256 less channels
# yaml_path = "/home/elhamod/projects/taming-transformers/logs/2022-10-12T23-03-58_Phylo-VQVAE256img-afterhyperp-ch64/configs/2022-10-12T23-03-58-project.yaml"
# ckpt_path = "/home/elhamod/projects/taming-transformers/logs/2022-10-12T23-03-58_Phylo-VQVAE256img-afterhyperp-ch64/checkpoints/last.ckpt"
# 256 no pass through
# yaml_path = "/home/elhamod/projects/taming-transformers/logs/2022-10-12T23-04-18_Phylo-VQVAE256img-afterhyperp-nopassthrough/configs/2022-10-12T23-04-18-project.yaml"
# ckpt_path = "/home/elhamod/projects/taming-transformers/logs/2022-10-12T23-04-18_Phylo-VQVAE256img-afterhyperp-nopassthrough/checkpoints/last.ckpt"

# 256 without phylo
# yaml_path = "/home/elhamod/projects/taming-transformers/logs/2022-10-12T10-08-53_Phylo-VQVAE256img-afterhyperp-withoutphylo-round2/configs/2022-10-12T10-08-53-project.yaml"
# ckpt_path = "/home/elhamod/projects/taming-transformers/logs/2022-10-12T10-08-53_Phylo-VQVAE256img-afterhyperp-withoutphylo-round2/checkpoints/last.ckpt"

# # # 512 less channels
# yaml_path = "/home/elhamod/projects/taming-transformers/logs/2022-10-14T11-39-30_Phylo-VQVAE512img-afterhyperp/configs/2022-10-14T11-39-30-project.yaml"
# ckpt_path = "/home/elhamod/projects/taming-transformers/logs/2022-10-14T11-39-30_Phylo-VQVAE512img-afterhyperp/checkpoints/last.ckpt"

# # 256 combined
# yaml_path = "/home/elhamod/projects/taming-transformers/logs/2022-10-14T00-55-06_Phylo-VQVAE256img-afterhyperp-combined/configs/2022-10-14T00-55-06-project.yaml"
# ckpt_path = "/home/elhamod/projects/taming-transformers/logs/2022-10-14T00-55-06_Phylo-VQVAE256img-afterhyperp-combined/checkpoints/last.ckpt"

# # 512 combined no passthrough
# yaml_path = "/home/elhamod/projects/taming-transformers/logs/2022-10-14T11-39-39_Phylo-VQVAE512img-afterhyperp/configs/2022-10-14T11-39-39-project.yaml"
# ckpt_path = "/home/elhamod/projects/taming-transformers/logs/2022-10-14T11-39-39_Phylo-VQVAE512img-afterhyperp/checkpoints/last.ckpt"


# #256 combined 36ch
# yaml_path = "/home/elhamod/projects/taming-transformers/logs/2022-10-16T00-21-41_Phylo-VQVAE/configs/2022-10-16T00-21-41-project.yaml"
# ckpt_path = "/home/elhamod/projects/taming-transformers/logs/2022-10-16T00-21-41_Phylo-VQVAE/checkpoints/last.ckpt"

# # 256 combined nopass
# yaml_path = "/home/elhamod/projects/taming-transformers/logs/2022-10-14T10-40-42_Phylo-VQVAE256img-afterhyperp-combined/configs/2022-10-14T10-40-42-project.yaml"
# ckpt_path = "/home/elhamod/projects/taming-transformers/logs/2022-10-14T10-40-42_Phylo-VQVAE256img-afterhyperp-combined/checkpoints/last.ckpt"

# 4cbs
yaml_path = "/home/elhamod/projects/taming-transformers/logs/2022-10-17T12-20-24_Phylo-VQVAE256img-afterhyperp-combined-4cbperlevel/configs/2022-10-17T12-20-24-project.yaml"
ckpt_path = "/home/elhamod/projects/taming-transformers/logs/2022-10-17T12-20-24_Phylo-VQVAE256img-afterhyperp-combined-4cbperlevel/checkpoints/last.ckpt"


# # 46ch
# yaml_path = "/home/elhamod/projects/taming-transformers/logs/2022-10-14T00-55-06_Phylo-VQVAE256img-afterhyperp-combined/configs/2022-10-14T00-55-06-project.yaml"
# ckpt_path = "/home/elhamod/projects/taming-transformers/logs/2022-10-14T00-55-06_Phylo-VQVAE256img-afterhyperp-combined/checkpoints/last.ckpt"

DEVICE=0

num_workers = 8
batch_size = 5 

size= 256
file_list_path = "/home/elhamod/data/Fish/taming_transforms_fish_train_padded_256.txt"

# size= 512
# file_list_path = "/home/elhamod/data/Fish/taming_transforms_fish_train_padded_512.txt"

# file_list_path = "/home/elhamod/data/Fish/taming_transforms_fish_test_small.txt" # Just for test


# imagepath="/home/elhamod/data/Fish/train_padded_256/Notropis percobromus/INHS_FISH_76701.jpg"
imagepath= "/home/elhamod/data/Fish/train_padded_256/Notropis percobromus/INHS_FISH_76701.jpg"

cummulative = False

plot_diff = False

# which_code_locations = [(0,0)]
which_phylo_levels=range(4)#[0] #range(4)
which_codebook_per_level=range(4)#[0] #range(8)

####################

class Clone_manger():
    def __init__(self, cummulative, embedding, codes):
        self.cummulative = cummulative
        self.embedding = embedding

        # if not cummulative:
        if cummulative:
        #         self.clone = torch.clone(embedding)
        # else:
            self.clone_list = []
            for code_index in codes:
                self.clone_list.append(torch.clone(embedding))

    def get_embedding(self, index=None):
        assert (index is not None) or (not self.cummulative)
        if not self.cummulative:
            return torch.clone(self.embedding)
        else:
            return self.clone_list[index]

@torch.no_grad()
def main():

    # Load model
    config = load_config(yaml_path, display=False)
    model = load_phylovqvae(config, ckpt_path=ckpt_path).to(DEVICE)
    model.set_test_chkpt_path(ckpt_path)

    # load image
    dataset = CustomDataset(size, file_list_path, add_labels=True)
    dataloader = DataLoader(dataset.data, batch_size=batch_size, num_workers=num_workers, collate_fn=custom_collate)
    processed_img = torch.from_numpy(dataloader.dataset.preprocess_image(imagepath)).unsqueeze(0)
    processed_img = processed_img.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format)
    processed_img = processed_img.float()

    # get output
    dec_image, _, _, in_out_disentangler = model(processed_img.to(DEVICE))
    q_phylo_output = in_out_disentangler[CONSTANTS.QUANTIZED_PHYLO_OUTPUT]

    converter = Embedding_Code_converter(model.phylo_disentangler.quantize.get_codebook_entry_index, model.phylo_disentangler.quantize.embedding, q_phylo_output[0, :, :, :].shape)
    all_code_indices = converter.get_phylo_codes(q_phylo_output[0, :, :, :].unsqueeze(0), verify=False)
    all_codes_reverse_reshaped = converter.get_phylo_embeddings(all_code_indices, verify=False)

    dec_image_reversed, _, _, _ = model(processed_img.to(DEVICE), overriding_quant=all_codes_reverse_reshaped)
    

    #TODO: why cant I move these globally?!?!?!?
    which_codes=[]#[0,10,20,30]#[0,1]# [0]

    if len(which_codes)==0:
        which_codes = range(model.phylo_disentangler.n_embed)


    
    clone_manager = Clone_manger(cummulative, all_codes_reverse_reshaped, which_codes)

    for level in which_phylo_levels:
        for code_location in which_codebook_per_level:
            generated_imgs = [dec_image, dec_image_reversed]

            # if not cummulative:
            #     all_codes_reverse_reshaped_clone = torch.clone(all_codes_reverse_reshaped)
            # else:
            #     all_codes_reverse_reshaped_clone_list = []
            #     for code_index in which_codes:
            #         all_codes_reverse_reshaped_clone_list.append(torch.clone(all_codes_reverse_reshaped))
            
                
                
            for code_index in tqdm(which_codes):
                # all_code_indices[code_location] = code_index
                # if cummulative:
                #     all_codes_reverse_reshaped_clone = all_codes_reverse_reshaped_clone_list[code_index]
                all_codes_reverse_reshaped_clone = clone_manager.get_embedding(code_index)
                all_codes_reverse_reshaped_clone[0, :, code_location, level] = model.phylo_disentangler.quantize.embedding(torch.tensor([code_index]).to(all_codes_reverse_reshaped.device))

                dec_image_new, _, _, _ = model(processed_img.to(DEVICE), overriding_quant=all_codes_reverse_reshaped_clone)
                
                if plot_diff:
                    generated_imgs.append(dec_image_new - dec_image)
                else:
                    generated_imgs.append(dec_image_new)
            
            generated_imgs = torch.cat(generated_imgs, dim=0)
            save_image_grid(generated_imgs, ckpt_path, subfolder="codebook_grid-cumulative{}".format(cummulative), postfix="level{}-location{}".format(level, code_location))

        

if __name__ == "__main__":
    main()