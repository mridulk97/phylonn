#!/bin/bash

#SBATCH --account=mabrownlab
#SBATCH --partition=dgx_normal_q
#SBATCH --time=0-15:00:00 
#SBATCH --gres=gpu:3
#SBATCH --nodes=1 --ntasks-per-node=1 --cpus-per-task=8
#SBATCH -o ./SLURM/slurm-%j.out


echo start load env and run python

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

# python main.py --name Phylo-VQVAE --base configs/custom_vqgan-256emb-512img-phylo-vqvae-afterhyperp.yaml -t True --gpus 0, #1 # crashes... not enough memory?!
# python main.py --name Phylo-VQVAE --base configs/custom_vqgan-256emb-512img-phylo-vqvae.yaml -t True --gpus 0, #1
# python main.py --name Phylo-VQVAE --base configs/custom_vqgan-256emb-512img-phylo-vqvae-phyloloss.yaml -t True --gpus 0,
python main.py --name LSF2-VQVAE --postfix 1024emb-256img_imagenetmean_noaug --base configs/custom_vqgan-1024emb-256img_imagenetmean_noaug-batch5-LSF2-vqvae.yaml -t True --gpus 0,1,2

exit;




# Run these for the dataset you want before updating the custom_vqgan.yaml file and then running this script
# find /home/elhamod/data/Fish/test_padded_512 -name "*.???" > /home/elhamod/data/Fish/taming_transforms_fish_test_padded_512.txt
# find /home/elhamod/data/Fish/train_padded_512 -name "*.???" > /home/elhamod/data/Fish/taming_transforms_fish_train_padded_512.txt

# find /home/elhamod/data/Fish/test_padded_256 -name "*.???" > /home/elhamod/data/Fish/taming_transforms_fish_test_padded_256.txt
# find /home/elhamod/data/Fish/train_padded_256 -name "*.???" > /home/elhamod/data/Fish/taming_transforms_fish_train_padded_256.txt