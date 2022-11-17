#!/bin/bash

#SBATCH --account=ml4science
#SBATCH --partition=dgx_normal_q
#SBATCH --time=0-10:00:00 
#SBATCH --gres=gpu:3
#SBATCH --nodes=1 --ntasks-per-node=1 --cpus-per-task=24
#SBATCH -o /fastscratch/elhamod/projects/taming-transformers/SLURM/slurm-%j.out
##########SBATCH -o ./SLURM/slurm-%j.out

# TODO: there is a bug. for some reason I need to reset again here.
module reset
module load Anaconda3/2020.11
source activate taming3 
module reset
source activate taming3 
which python

# python main.py --prefix /fastscratch/elhamod --name Phylo-VQVAE-phase6 --postfix 256img-phase6 --base /home/elhamod/projects/taming-transformers/configs/custom_vqgan-256emb-256img-phylo-vqvae-phase6.yaml -t True --gpus 0,
# python main.py --prefix /fastscratch/elhamod --name Phylo-VQVAE-phase6 --postfix 256img-phase6-phyloalone --base /home/elhamod/projects/taming-transformers/configs/custom_vqgan-256emb-256img-phylo-vqvae-phase6-phyloalone.yaml -t True --gpus 0,
# python main.py --prefix /fastscratch/elhamod --name Phylo-VQVAE-phase6 --postfix 256img-phase6-speciesout --base /home/elhamod/projects/taming-transformers/configs/custom_vqgan-256emb-256img-phylo-vqvae-phase6-speciesout.yaml -t True --gpus 0,
# python main.py --prefix /fastscratch/elhamod --name Phylo-VQVAE-phase6 --postfix 256img-phase6-phylohigh --base /home/elhamod/projects/taming-transformers/configs/custom_vqgan-256emb-256img-phylo-vqvae-phase6-phylohigh.yaml -t True --gpus 0,
# python main.py --prefix /fastscratch/elhamod --name Phylo-VQVAE-phase6 --postfix 256img-phase6-morech --base /home/elhamod/projects/taming-transformers/configs/custom_vqgan-256emb-256img-phylo-vqvae-phase6-morech.yaml -t True --gpus 0,
# python main.py --prefix /fastscratch/elhamod --name Phylo-VQVAE-phase6 --postfix 256img-phase6-phyloquantize --base /home/elhamod/projects/taming-transformers/configs/custom_vqgan-256emb-256img-phylo-vqvae-phase6-phyloplusquant.yaml -t True --gpus 0,
# python main.py --prefix /fastscratch/elhamod --name Phylo-VQVAE-phase6 --postfix 256img-phase6-phyloquantizerec --base /home/elhamod/projects/taming-transformers/configs/custom_vqgan-256emb-256img-phylo-vqvae-phase6-phyloplusquantrec.yaml -t True --gpus 0,
# python main.py --prefix /fastscratch/elhamod --name Phylo-VQVAE-phase6 --postfix 256img-phase6-noorth --base /home/elhamod/projects/taming-transformers/configs/custom_vqgan-256emb-256img-phylo-vqvae-phase6-noorth.yaml -t True --gpus 0,
# python main.py --prefix /fastscratch/elhamod --name Phylo-VQVAE-phase6 --postfix 256img-phase6-highortho --base /home/elhamod/projects/taming-transformers/configs/custom_vqgan-256emb-256img-phylo-vqvae-phase6-highortho.yaml -t True --gpus 0,
# python main.py --prefix /fastscratch/elhamod --name Phylo-VQVAE-phase6 --postfix 256img-phase6-lowerlr --base /home/elhamod/projects/taming-transformers/configs/custom_vqgan-256emb-256img-phylo-vqvae-phase6-lowerlr.yaml -t True --gpus 0,
python main.py --prefix /fastscratch/elhamod --name Phylo-VQVAE-phase6 --postfix 256img-phase6-fixoptimization --base /home/elhamod/projects/taming-transformers/configs/custom_vqgan-256emb-256img-phylo-vqvae-phase6-autooptimization.yaml -t True --gpus 0,


exit;





# Run these for the dataset you want before updating the custom_vqgan.yaml file and then running this script
# find /home/elhamod/data/Fish/test_padded_512 -name "*.???" > /home/elhamod/data/Fish/taming_transforms_fish_test_padded_512.txt
# find /home/elhamod/data/Fish/train_padded_512 -name "*.???" > /home/elhamod/data/Fish/taming_transforms_fish_train_padded_512.txt

# find /home/elhamod/data/Fish/test_padded_256 -name "*.???" > /home/elhamod/data/Fish/taming_transforms_fish_test_padded_256.txt
# find /home/elhamod/data/Fish/train_padded_256 -name "*.???" > /home/elhamod/data/Fish/taming_transforms_fish_train_padded_256.txt