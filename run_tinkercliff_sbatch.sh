#!/bin/bash
#SBATCH -J VQGAN
#SBATCH --account=ml4science
#SBATCH --partition=dgx_normal_q
#SBATCH --nodes=1 --ntasks-per-node=1 --cpus-per-task=8 # this requests 1 node, 1 core. 
#SBATCH --time=0-12:00:00 # 10 minutes
#SBATCH --gres=gpu:3

module reset

module load Anaconda3/2020.11

source activate taming
module reset
which python

# python main.py --name CW-VQVAE --postfix 256img--tinker --base configs/custom_vqgan-256img-phylo-vqvae.yaml -t True --gpus 0,
# python main.py --name CW-VQVAE --postfix 256img-tinker-cw-model-all --base configs/custom_vqgan-256img-cw-module.yaml -t True --gpus 0,1,2,

# python main.py --name CW-VQGAN --postfix testing_merge_code --base configs/custom_vqgan-256img-cw-module.yaml -t True --gpus 0,

#cw-baseline-test
# python main.py --name CW-VQVAE --postfix 256img-tinker-cw-baseline --base configs/cw_baseline_256img.yaml -t True --gpus 0,1,2

#transformers
# python main.py --name CW-VQGAN-transformer --postfix new-tinker-latest --base configs/cw_transformer_latest_tinker_model.yaml -t True --gpus 0,1,2,3,4
# python main.py --name CUB-VQGAN --postfix cub_segmented_white_padding_bb_crop --base configs/cub_segmented_white_padding_bb_crop.yaml -t True --gpus 0,1,2,3,4,

python main.py --name CUB-PhyloNN --postfix 128_codes_seg_white_aug_conv_128_phylochannels_l2_0.3 --base configs/phyloNN_256_codes.yaml -t True --gpus 0,1,2,

# python main.py --name CUB-PhyloNN-debug --postfix conv_debug --base configs/conv_debug.yaml -t True --gpus 0,



exit;