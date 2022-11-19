#!/bin/bash

#SBATCH --account=ml4science
#SBATCH --partition=v100_normal_q
#SBATCH --time=0-24:00:00 
#SBATCH --gres=gpu:1 
#SBATCH -o /fastscratch/elhamod/projects/taming-transformers/SLURM/slurm-%j.out



### IMPORTANT:
## For some reason this only runs from:
# [elhamod@inf083 taming-transformers]$ sbatch hyperp/hyperp_phylo-VQVAE-infer.sh
# Make sure you are at that directory and using relative path for the sh file.
### MAKESURE you have not loaded taming3 env!

module reset

module load Anaconda3/2020.11

source activate taming3 

which python

which python3

#python -m 
# wandb agent mndhamod/Phylo-VQVAE/62lfnhd1
wandb agent mndhamod/Phylo-VQVAE/9q6l2hyv

# To create a sweep:
# wandb sweep --project Phylo-VQVAE hyperp/hyperp_bayes_nested.yaml 
# wandb sweep --project Phylo-VQVAE hyperp/hyperp_bayes_phyloloss_nested.yaml
# wandb sweep --project Phylo-VQVAE hyperp/hyperp_grid_phyloloss_codebookperlevel_nested.yaml
# wandb sweep --project Phylo-VQVAE hyperp/hyperp_grid_phyloloss_numcodebooks_nested.yaml
# wandb sweep --project Phylo-VQVAE hyperp/hyperp_grid_phyloloss_numphylochannels_nested.yaml
# wandb sweep --project Phylo-VQVAE hyperp/hyperp_grid_phyloloss_numtotalchannels_nested.yaml

exit;