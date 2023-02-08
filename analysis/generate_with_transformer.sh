#!/bin/bash

#SBATCH --account=imageomics-biosci
#SBATCH --partition=a100_normal_q  #a100_normal_q
#SBATCH --time=0-30:00:00 
#SBATCH --gres=gpu:1
#SBATCH --nodes=1 --ntasks-per-node=1 --cpus-per-task=8
#SBATCH -o /fastscratch/elhamod/projects/phylonn/SLURM/slurm-%j.out

##########SBATCH -o ./SLURM/slurm-%j.out


# echo start load env and run python

module reset

module load Anaconda3/2020.11
# TODO: there is a bug. for some reason I need to reset again here.
source activate taming3 
module reset
source activate taming3 

which python

python analysis/generate_with_transformer.py 

exit;