from taming.loading_utils import load_config, load_phylovqvae
from taming.data.custom import CustomTest as CustomDataset
from taming.data.utils import custom_collate

from torch.utils.data import DataLoader
from pytorch_lightning.trainer import Trainer
import torch




###################

# parameters
# best 256 with phylo
# yaml_path = "/home/elhamod/projects/taming-transformers/logs/2022-10-07T12-09-20_Phylo-VQVAE256img-afterhyperp/configs/2022-10-07T12-09-20-project.yaml"
# ckpt_path = "/home/elhamod/projects/taming-transformers/logs/2022-10-07T12-09-20_Phylo-VQVAE256img-afterhyperp/checkpoints/last.ckpt"

# 256 without phylo (TODO: maybe rerun this with best hyperp?)
# yaml_path = "/home/elhamod/projects/taming-transformers/logs/256img swap conv and silu 2022-10-05T08-52-16_Phylo-VQVAE-test/configs/2022-10-05T08-52-16-project.yaml"
# ckpt_path = "/home/elhamod/projects/taming-transformers/logs/256img swap conv and silu 2022-10-05T08-52-16_Phylo-VQVAE-test/checkpoints/last.ckpt"

# # 256 with korthogonality
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

# # 512 less channels
# yaml_path = "/home/elhamod/projects/taming-transformers/logs/2022-10-14T11-39-30_Phylo-VQVAE512img-afterhyperp/configs/2022-10-14T11-39-30-project.yaml"
# ckpt_path = "/home/elhamod/projects/taming-transformers/logs/2022-10-14T11-39-30_Phylo-VQVAE512img-afterhyperp/checkpoints/last.ckpt"

# # 256 combined
# yaml_path = "/home/elhamod/projects/taming-transformers/logs/2022-10-14T00-55-06_Phylo-VQVAE256img-afterhyperp-combined/configs/2022-10-14T00-55-06-project.yaml"
# ckpt_path = "/home/elhamod/projects/taming-transformers/logs/2022-10-14T00-55-06_Phylo-VQVAE256img-afterhyperp-combined/checkpoints/last.ckpt"


# 256 combined nopass
yaml_path = "/home/elhamod/projects/taming-transformers/logs/2022-10-14T10-40-42_Phylo-VQVAE256img-afterhyperp-combined/configs/2022-10-14T10-40-42-project.yaml"
ckpt_path = "/home/elhamod/projects/taming-transformers/logs/2022-10-14T10-40-42_Phylo-VQVAE256img-afterhyperp-combined/checkpoints/last.ckpt"



DEVICE=0

num_workers = 8
batch_size = 5 


size= 256
file_list_path = "/home/elhamod/data/Fish/taming_transforms_fish_train_padded_256.txt"

# size= 512
# file_list_path = "/home/elhamod/data/Fish/taming_transforms_fish_train_padded_512.txt"

# file_list_path = "/home/elhamod/data/Fish/taming_transforms_fish_test_small.txt" # Just for test


####################

@torch.no_grad()
def main():
    config = load_config(yaml_path, display=False)
    model = load_phylovqvae(config, ckpt_path=ckpt_path).to(DEVICE)
    model.set_test_chkpt_path(ckpt_path)

    dataset = CustomDataset(size, file_list_path, add_labels=True)
    dataloader = DataLoader(dataset.data, batch_size=batch_size, num_workers=num_workers, collate_fn=custom_collate)

    trainer = Trainer(distributed_backend='ddp', gpus='0,')
    test_measures = trainer.test(model, dataloader)

    print('test_measures', test_measures)

if __name__ == "__main__":
    main()