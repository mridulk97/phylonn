#!/bin/bash
#SBATCH -J VQGAN
#SBATCH --account=mabrownlab
#SBATCH --partition=dgx_normal_q
#SBATCH --nodes=1 --ntasks-per-node=1 --cpus-per-task=8 # this requests 1 node, 1 core. 
#SBATCH --time=0-04:00:00 # 10 minutes
#SBATCH --gres=gpu:4

module reset

module load Anaconda3/2020.11

source activate taming
module reset
which python

# python main.py --name CW-VQVAE --postfix 256img--tinker --base configs/custom_vqgan-256img-phylo-vqvae.yaml -t True --gpus 0,
# python main.py --name CW-VQVAE --postfix cw_old_model_data_test --base configs/custom_vqgan-256img-cw-module.yaml -t True --gpus 0,1,

# python main.py --name CW-VQGAN --postfix testing_merge_code --base configs/custom_vqgan-256img-cw-module.yaml -t True --gpus 0,

#cw-baseline-test
# python main.py --name CW-VQVAE --postfix 256img-tinker-cw-baseline --base configs/cw_baseline_256img.yaml -t True --gpus 0,1,2

#transformers
# python main.py --name CW-VQGAN-transformer --postfix new-tinker-latest --base configs/cw_transformer_latest_tinker_model.yaml -t True --gpus 0,1,2,3,4
# python main.py --name CW-VQGAN-transformer --postfix testing_merge_code_transformers --base configs/cw_transformer_latest_tinker_model.yaml -t True --gpus 0,

# ## CUB
# python main.py --name CUB-VQGAN --postfix cub_n_embed_1024_epoch_771_onwards --base configs/cub_vqgan_256_test.yaml -t True --gpus 0,1,

# CUB transformer
# python main.py --name CUB-VQGAN-transformer --postfix cub_n_embed_1024_transformer_test --base configs/cub_vqgan_transformer.yaml -t True --gpus 0,1,2,3,

#CUB - CW
# python main.py --name CUB-MODELS --postfix cub_cw_module_n_embed_1024_epoch_467 --base configs/cub_cw_module_256.yaml -t True --gpus 0,1,

## CUB - segmented
# python main.py --name CUB-VQGAN --postfix cub_segmented --base configs/cub_vqgan_256_segmented.yaml -t True --gpus 0,1,2,3,4,

## CUB - segmented transformer
# python main.py --name CUB-VQGAN-transformer --postfix cub_bb_crop_aug_imagenet_padding_epoch_561_transformer --base configs/cub_vqgan_256_aug_transformer.yaml -t True --gpus 0,1,

# python main.py --name CUB-VQGAN-transformer --postfix cub_bb_crop_aug_imagenet_padding_segmented_epoch_555 --base configs/cub_vqgan_256_aug_segmented_transformer.yaml -t True --gpus 0,1,

# python main.py --name CUB-VQGAN --postfix cub_bb_crop_augmentations_imagenet_padding_epoch_561_onwards --base configs/cub_vqgan_256_augmentations.yaml -t True --gpus 0,1,2,3,4,

# python main.py --name CUB-VQGAN --postfix cub_bb_crop_augmentations_imagenet_padding_segmented_epoch_555 --base configs/cub_vqgan_256_augmentations_segmented.yaml -t True --gpus 0,1,2,3,4,
python main.py --name CW-VQGAN-transformer --postfix cw_imagenet_no_aug_base_model_transformer --base configs/cw_transformer_cw_imagenet_no_aug.yaml -t True --gpus 0,1,2,3,

exit;