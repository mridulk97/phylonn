#!/bin/bash
#SBATCH -J VQVAE
#SBATCH --account=ml4science
#SBATCH --partition=v100_normal_q
#SBATCH --nodes=1 --ntasks-per-node=1 --cpus-per-task=8 # this requests 1 node, 1 core. 
#SBATCH --time=2-00:00:00 # 10 minutes
#SBATCH --gres=gpu:1

module reset

module load Anaconda3/2020.11
# module load gcc/8.2.0

source activate taming3
module reset
# source activate taming3
which python

# python main.py --name Phylo-VQVAE --postfix 256img-testrun-infer --base configs/custom_vqgan-256img-phylo-vqvae.yaml -t True --gpus 0,

# python main.py --name CW-VQVAE --postfix 256img-infer-base_VQGAN --base configs/custom_vqgan-256.yaml -t True --gpus 0,÷
python main.py --name CW-VQVAE --postfix 256img-infer-cw-model --base configs/custom_vqgan-256img-cw-module.yaml -t True --gpus 0,

# python main_cw_analysis.py --name CW-VQVAE --postfix 256img-infer-cw-debug --base configs/custom_vqgan-256img-cw-module.yaml -p True --gpus 0,

exit;