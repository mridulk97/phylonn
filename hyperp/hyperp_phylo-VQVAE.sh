#!/bin/bash

#SBATCH --account=ml4science
#SBATCH --partition=a100_normal_q
#SBATCH --time=2-00:00:00 
#SBATCH --gres=gpu:1 
#SBATCH --nodes=1 --ntasks-per-node=1 --cpus-per-task=8
#SBATCH -o ./SLURM/slurm-%j.out

module reset

module load Anaconda3/2020.11
module load gcc/8.2.0

source activate taming3 

#python -m 
wandb agent mndhamod/Phylo-VQVAE/vm4v0kh3

# To create a sweep:
# wandb sweep --project Phylo-VQVAE hyperp/hyperp_bayes_nested.yaml 

exit;