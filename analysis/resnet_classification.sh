#!/bin/bash

#SBATCH --account=imageomics-biosci
#SBATCH --partition=dgx_normal_q  #a100_normal_q
#SBATCH --time=0-30:00:00 
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

# python analysis/resnet_classification.py --config /home/elhamod/projects/taming-transformers/analysis/configs/resnet_classification.yaml
# python analysis/resnet_classification.py --config /home/elhamod/projects/taming-transformers/analysis/configs/resnet_classification-lvl1.yaml
# python analysis/resnet_classification.py --config /home/elhamod/projects/taming-transformers/analysis/configs/resnet_classification-lvl2.yaml
python analysis/resnet_classification.py --config /home/elhamod/projects/taming-transformers/analysis/configs/resnet_classification-lvl0.yaml

exit;