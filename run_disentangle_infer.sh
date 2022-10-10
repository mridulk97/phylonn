#!/bin/bash

#SBATCH --account=ml4science
#SBATCH --partition=v100_normal_q
#SBATCH --time=1-00:00:00 
#SBATCH --gres=gpu:1
#SBATCH -o ./SLURM/slurm-%j.out


echo start load env and run python

module reset

module load Anaconda3/2020.11
# module load gcc/8.2.0

source activate taming3 

# python main.py --name Phylo-VQVAE --base configs/custom_vqgan-256img-phylo-vqvae.yaml -t True --gpus 0,
# python main.py --name Phylo-VQVAE --base configs/custom_vqgan-256img-phylo-vqvae-phyloloss.yaml -t True --gpus 0,
python main.py --name Phylo-VQVAE-test --postfix 256img-afterhyperp --base configs/custom_vqgan-256img-phylo-vqvae-phyloloss-afterhyperp.yaml -t True --gpus 0,

exit;




# Run these for the dataset you want before updating the custom_vqgan.yaml file and then running this script
# find /home/elhamod/data/Fish/test_padded_512 -name "*.???" > /home/elhamod/data/Fish/taming_transforms_fish_test_padded_512.txt
# find /home/elhamod/data/Fish/train_padded_512 -name "*.???" > /home/elhamod/data/Fish/taming_transforms_fish_train_padded_512.txt

# find /home/elhamod/data/Fish/test_padded_256 -name "*.???" > /home/elhamod/data/Fish/taming_transforms_fish_test_padded_256.txt
# find /home/elhamod/data/Fish/train_padded_256 -name "*.???" > /home/elhamod/data/Fish/taming_transforms_fish_train_padded_256.txt