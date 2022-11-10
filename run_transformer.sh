#!/bin/bash

#SBATCH --account=ml4science
#SBATCH --partition=dgx_normal_q  #a100_normal_q
#SBATCH --time=2-00:00:00 
#SBATCH --gres=gpu:1
#SBATCH --nodes=1 --ntasks-per-node=1 --cpus-per-task=8
#SBATCH -o /fastscratch/elhamod/projects/taming-transformers/SLURM/slurm-%j.out

##########SBATCH -o ./SLURM/slurm-%j.out


# echo start load env and run python

module reset

module load Anaconda3/2020.11
# module load gcc/8.2.0

# module reset

# module load Anaconda3/2020.11

# TODO: there is a bug. for some reason I need to reset again here.
source activate taming3 
module reset
source activate taming3 

which python

# python main.py --prefix /fastscratch/elhamod --name Phylo-VQVAE-transformer --postfix 256img-phase4-transformer-promising --base configs/phylo_vqgan_transformer.yaml -t True --gpus 0, 

# python main.py --prefix /fastscratch/elhamod --name Phylo-VQVAE-transformer --postfix 512img-phase4-cyclical-largerspace --base configs/phylo_vqgan_transformer-512-cyclical-largerspace.yaml -t True --gpus 0, 

# python main.py --prefix /fastscratch/elhamod --name Phylo-VQVAE-transformer --postfix 256img-phase4-originalVQGAN --base configs/original_VQGAN_transformer_fish.yaml -t True --gpus 0, 

# python main.py --prefix /fastscratch/elhamod --name Phylo-VQVAE-transformer --postfix 256img-phase4-level0 --base configs/phylo_vqgan_transformer-level0.yaml -t True --gpus 0, 
# python main.py --prefix /fastscratch/elhamod --name Phylo-VQVAE-transformer --postfix 256img-phase4-level1 --base configs/phylo_vqgan_transformer-level1.yaml -t True --gpus 0, 
python main.py --prefix /fastscratch/elhamod --name Phylo-VQVAE-transformer --postfix 256img-phase4-level2 --base configs/phylo_vqgan_transformer-level2.yaml -t True --gpus 0, 


exit;




# Run these for the dataset you want before updating the custom_vqgan.yaml file and then running this script
# find /home/elhamod/data/Fish/test_padded_512 -name "*.???" > /home/elhamod/data/Fish/taming_transforms_fish_test_padded_512.txt
# find /home/elhamod/data/Fish/train_padded_512 -name "*.???" > /home/elhamod/data/Fish/taming_transforms_fish_train_padded_512.txt

# find /home/elhamod/data/Fish/test_padded_256 -name "*.???" > /home/elhamod/data/Fish/taming_transforms_fish_test_padded_256.txt
# find /home/elhamod/data/Fish/train_padded_256 -name "*.???" > /home/elhamod/data/Fish/taming_transforms_fish_train_padded_256.txt