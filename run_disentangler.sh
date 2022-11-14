#!/bin/bash

#SBATCH --account=ml4science
#SBATCH --partition=a100_normal_q
#SBATCH --time=2-00:00:00 
#SBATCH --gres=gpu:1
#SBATCH --nodes=1 --ntasks-per-node=1 --cpus-per-task=8
#SBATCH -o /fastscratch/elhamod/projects/taming-transformers/SLURM/slurm-%j.out

##########SBATCH -o ./SLURM/slurm-%j.out

# TODO: there is a bug. for some reason I need to reset again here.
module reset
module load Anaconda3/2020.11
source activate taming3 
module reset
source activate taming3 
which python

python main.py --prefix /fastscratch/elhamod --name Phylo-VQVAE --postfix 512img-phase5-lowerphyloweight --base configs/custom_vqgan-256emb-256img-phylo-vqvae-phase5-lowerphyloweight.yaml -t True --gpus 0, 

exit;




# Run these for the dataset you want before updating the custom_vqgan.yaml file and then running this script
# find /home/elhamod/data/Fish/test_padded_512 -name "*.???" > /home/elhamod/data/Fish/taming_transforms_fish_test_padded_512.txt
# find /home/elhamod/data/Fish/train_padded_512 -name "*.???" > /home/elhamod/data/Fish/taming_transforms_fish_train_padded_512.txt

# find /home/elhamod/data/Fish/test_padded_256 -name "*.???" > /home/elhamod/data/Fish/taming_transforms_fish_test_padded_256.txt
# find /home/elhamod/data/Fish/train_padded_256 -name "*.???" > /home/elhamod/data/Fish/taming_transforms_fish_train_padded_256.txt