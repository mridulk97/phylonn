#!/bin/bash

#SBATCH --account=ml4science
#SBATCH --partition=v100_normal_q
#SBATCH --time=2-00:00:00 
#SBATCH --gres=gpu:1
#SBATCH -o ./SLURM/slurm-%j.out


echo start load env and run python

module reset

module load Anaconda3/2020.11
# module load gcc/8.2.0

source activate taming3 

which python

# python main.py --name Phylo-VQVAE --base configs/custom_vqgan-256img-phylo-vqvae.yaml -t True --gpus 0,
# python main.py --name Phylo-VQVAE --base configs/custom_vqgan-256img-phylo-vqvae-phyloloss.yaml -t True --gpus 0,
# python main.py --name Phylo-VQVAE-test --postfix 256img-afterhyperp-withoutphyloss-round2 --base configs/custom_vqgan-256img-phylo-vqvae-phyloloss-afterhyperp.yaml -t True --gpus 0,
# python main.py --name Phylo-VQVAE-test --postfix 256img-afterhyperp-roundtest --base configs/custom_vqgan-256img-phylo-vqvae-withoutphyloloss-afterhyperp.yaml -t True --gpus 0,

# python main.py --name Phylo-VQVAE --postfix 256img-afterhyperp-ch64 --base configs/custom_vqgan-256emb-512img-phylo-vqvae-phyloloss-afterhyperp-ch64.yaml -t True --gpus 0,
# python main.py --name Phylo-VQVAE --postfix 256img-afterhyperp-nopassthrough --base configs/custom_vqgan-256emb-512img-phylo-vqvae-phyloloss-afterhyperp-nopassthrough.yaml -t True --gpus 0,
# python main.py --name Phylo-VQVAE --postfix 256img-afterhyperp-addanticlassloss --base configs/custom_vqgan-256emb-512img-phylo-vqvae-phyloloss-afterhyperp-anticlassloss.yaml -t True --gpus 0,
# python main.py --name Phylo-VQVAE --postfix 256img-afterhyperp-kernelorthogonality --base configs/custom_vqgan-256emb-512img-phylo-vqvae-phyloloss-afterhyperp-kernelorthogonality.yaml -t True --gpus 0,

# python main.py --name Phylo-VQVAE --postfix 256img-afterhyperp-combined --base configs/custom_vqgan-256emb-512img-phylo-vqvae-phyloloss-afterhyperp-ch64.yaml -t True --gpus 0,
# python main.py --name Phylo-VQVAE --postfix 256img-afterhyperp-combined --base configs/custom_vqgan-256emb-512img-phylo-vqvae-phyloloss-afterhyperp-combination-nopassthrough.yaml -t True --gpus 0,
# python main.py --name Phylo-VQVAE --postfix 256img-afterhyperp-combined-4cbperlevel --base configs/custom_vqgan-256emb-512img-phylo-vqvae-phyloloss-afterhyperp-combination-nopassthrough-4cbperlevel.yaml -t True --gpus 0,
python main.py --name Phylo-VQVAE --postfix 256img-afterhyperp-combined-anticlass --base configs/custom_vqgan-256emb-512img-phylo-vqvae-phyloloss-afterhyperp-anticlassloss.yaml -t True --gpus 0,


# resume
# python main.py --resume /home/elhamod/projects/taming-transformers/logs/2022-10-14T10-40-42_Phylo-VQVAE256img-afterhyperp-combined --postfix 256img-afterhyperp-combined --base configs/custom_vqgan-256emb-512img-phylo-vqvae-phyloloss-afterhyperp-combination-nopassthrough.yaml -t True --gpus 0,



exit;




# Run these for the dataset you want before updating the custom_vqgan.yaml file and then running this script
# find /home/elhamod/data/Fish/test_padded_512 -name "*.???" > /home/elhamod/data/Fish/taming_transforms_fish_test_padded_512.txt
# find /home/elhamod/data/Fish/train_padded_512 -name "*.???" > /home/elhamod/data/Fish/taming_transforms_fish_train_padded_512.txt

# find /home/elhamod/data/Fish/test_padded_256 -name "*.???" > /home/elhamod/data/Fish/taming_transforms_fish_test_padded_256.txt
# find /home/elhamod/data/Fish/train_padded_256 -name "*.???" > /home/elhamod/data/Fish/taming_transforms_fish_train_padded_256.txt