#!/bin/bash
#SBATCH -J VQVAE
#SBATCH --account=ml4science
#SBATCH --partition=dgx_normal_q
#SBATCH --nodes=1 --ntasks-per-node=1 --cpus-per-task=8 # this requests 1 node, 1 core. 
#SBATCH --time=1-00:00:00 # 10 minutes
#SBATCH --gres=gpu:5

module reset

module load Anaconda3/2020.11

source activate taming
module reset
which python

# python main.py --name CW-VQVAE --postfix 256img--tinker --base configs/custom_vqgan-256img-phylo-vqvae.yaml -t True --gpus 0,
# python main.py --name CW-VQVAE --postfix 256img-tinker-cw-model-all --base configs/custom_vqgan-256img-cw-module.yaml -t True --gpus 0,1,2,

# python main.py --name CW-VQVAE --postfix 256img--test --base configs/custom_vqgan-256img-cw-module.yaml -t True --gpus 0,

#cw-baseline-test
# python main.py --name CW-VQVAE --postfix 256img-tinker-cw-baseline --base configs/cw_baseline_256img.yaml -t True --gpus 0,1,2

#transformers
# python main.py --name CW-VQGAN-transformer --postfix new-tinker-latest --base configs/cw_transformer_latest_tinker_model.yaml -t True --gpus 0,1,2,3,4


exit;