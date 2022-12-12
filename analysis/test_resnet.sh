#!/bin/bash
#SBATCH -J VQVAE
#SBATCH --account=ml4science
#SBATCH --partition=dgx_normal_q
#SBATCH --nodes=1 --ntasks-per-node=1 --cpus-per-task=8 # this requests 1 node, 1 core. 
#SBATCH --time=0-01:00:00 # 10 minutes
#SBATCH --gres=gpu:1

##########SBATCH -o ./SLURM/slurm-%j.out


module reset

module load Anaconda3/2020.11

source activate taming
module reset
which python


python analysis/test_resnet.py
python analysis/test_resnet.py --config analysis/configs/test_resnet_level0.yaml
python analysis/test_resnet.py --config analysis/configs/test_resnet_level1.yaml
python analysis/test_resnet.py --config analysis/configs/test_resnet_level2.yaml


exit;