#!/bin/bash

#SBATCH --account=ml4science
#SBATCH --partition=a100_normal_q
#SBATCH --time=1-00:00:00 
#SBATCH --gres=gpu:1 
#SBATCH --nodes=1 --ntasks-per-node=1 --cpus-per-task=8
#SBATCH -o /fastscratch/elhamod/projects/taming-transformers/SLURM/slurm-%j.out

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

wandb agent mndhamod/Phylo-VQVAE-transformer/av4oob0x

# To create a sweep:
# wandb sweep --project Phylo-VQVAE-transformer hyperp/hyperp_bayes_phylo-transformer.yaml

exit;