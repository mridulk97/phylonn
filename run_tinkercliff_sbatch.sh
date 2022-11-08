#!/bin/bash
#SBATCH -J VQVAE
#SBATCH --account=ml4science
#SBATCH --partition=dgx_normal_q
#SBATCH --nodes=1 --ntasks-per-node=1 --cpus-per-task=8 # this requests 1 node, 1 core. 
#SBATCH --time=2-00:00:00 # 10 minutes
#SBATCH --gres=gpu:1

module reset

module load Anaconda3/2020.11

source activate taming3
module reset
which python

# python main.py --name CW-VQVAE --postfix 256img--tinker --base configs/custom_vqgan-256img-phylo-vqvae.yaml -t True --gpus 0,
python main.py --name CW-VQVAE --postfix 256img-tinker-cw-model --base configs/custom_vqgan-256img-cw-module.yaml -t True --gpus 0,

# python main.py --name CW-VQVAE --postfix 256img--tinker --base configs/custom_vqgan-256.yaml -t True --gpus 0,


exit;