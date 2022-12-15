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

#python -m 
# wandb agent mndhamod/Phylo-VQVAE/q58r7rik
# wandb agent mndhamod/Phylo-VQVAE/kih79ivcp
# wandb agent mndhamod/Phylo-VQVAE/wmpsdu5l
# wandb agent mndhamod/Phylo-VQVAE/o9w7bvqg
# wandb agent mndhamod/Phylo-VQVAE/9q6l2hyv
# wandb agent mndhamod/Phylo-VQVAE/z5a4bg1j
#  wandb agent mndhamod/Phylo-VQVAE/jzwqt7q2
wandb agent mndhamod/Phylo-VQVAE/pxuof5hm

# To create a sweep:
# wandb sweep --project Phylo-VQVAE hyperp/hyperp_bayes_nested.yaml 
# wandb sweep --project Phylo-VQVAE hyperp/hyperp_bayes_phyloloss_nested.yaml
# wandb sweep --project Phylo-VQVAE hyperp/hyperp_grid_phyloloss_codebookperlevel_nested.yaml
# wandb sweep --project Phylo-VQVAE hyperp/hyperp_grid_phyloloss_numcodebooks_nested.yaml
# wandb sweep --project Phylo-VQVAE hyperp/hyperp_grid_phyloloss_numphylochannels_nested.yaml
# wandb sweep --project Phylo-VQVAE hyperp/hyperp_bayes_morenonphyloch_nested.yaml

exit;