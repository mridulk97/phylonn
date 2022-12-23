#!/bin/bash

#SBATCH --account=ml4science
#SBATCH --partition=dgx_normal_q #a100_normal_q  #dgx_normal_q  #a100_normal_q
#SBATCH --time=2-00:00:00 
#SBATCH --gres=gpu:1
#SBATCH --nodes=1 --ntasks-per-node=1 --cpus-per-task=8
#SBATCH -o /fastscratch/elhamod/projects/taming-transformers/SLURM/slurm-%j.out

##########SBATCH -o ./SLURM/slurm-%j.out

# TODO: there is a bug. for some reason I need to reset again here.
module reset
module load Anaconda3/2020.11
source activate taming3 
module reset
source activate taming3 
which python

# python main.py --prefix /fastscratch/elhamod --name Phylo-VQVAE-transformer --postfix 256img-phase4-level3-finalstronger --base configs/phylo_vqgan_transformer-level3.yaml -t True --gpus 0, 
# python main.py --prefix /fastscratch/elhamod --name Phylo-VQVAE-transformer-phase6 --postfix 256img-phase6 --base /home/elhamod/projects/taming-transformers/configs/phylo_vqgan_transformer-phase6.yaml -t True --gpus 0, 
# python main.py --prefix /fastscratch/elhamod --name Phylo-VQVAE-transformer-phase6 --postfix 256img-phase6-less --base /home/elhamod/projects/taming-transformers/configs/phylo_vqgan_transformer-phase6-less.yaml -t True --gpus 0, 

# python main.py --prefix /fastscratch/elhamod --name Phylo-VQVAE-transformer-phase7 --postfix 256img-phase7-level3-idea3 --base /home/elhamod/projects/taming-transformers/configs/phylo_vqgan_transformer-phase7-level3-idea3.yaml -t True --gpus 0,
# python main.py --prefix /fastscratch/elhamod --name Phylo-VQVAE-transformer-phase7 --postfix 256img-phase7-level2-idea3 --base /home/elhamod/projects/taming-transformers/configs/phylo_vqgan_transformer-phase7-level2-idea3.yaml -t True --gpus 0,
# python main.py --prefix /fastscratch/elhamod --name Phylo-VQVAE-transformer-phase7 --postfix 256img-phase7-level1-idea3 --base /home/elhamod/projects/taming-transformers/configs/phylo_vqgan_transformer-phase7-level1-idea3.yaml -t True --gpus 0,
# python main.py --prefix /fastscratch/elhamod --name Phylo-VQVAE-transformer-phase7 --postfix 256img-phase7-level0-idea3 --base /home/elhamod/projects/taming-transformers/configs/phylo_vqgan_transformer-phase7-level0-idea3.yaml -t True --gpus 0,
# python main.py --prefix /fastscratch/elhamod --name Phylo-VQVAE-transformer-phase7 --postfix 256img-phase7-uncoditional-nonphylo --base /home/elhamod/projects/taming-transformers/configs/phylo_vqgan_transformer-phase7-uncoditional-nonphylo.yaml -t True --gpus 0,
# python main.py --prefix /fastscratch/elhamod --name Phylo-VQVAE-transformer-phase7 --postfix 256img-phase7-uncoditional --base /home/elhamod/projects/taming-transformers/configs/phylo_vqgan_transformer-phase7-uncoditional.yaml -t True --gpus 0,

# python main.py --prefix /fastscratch/elhamod --name Phylo-VQVAE-transformer-phase7 --postfix 256img-phase7-level3-partial_codes-morespace-samelayers-slower --base /home/elhamod/projects/taming-transformers/configs/phylo_vqgan_transformer-phase7-level3-morespace-samelayers-slower.yaml -t True --gpus 0,

python main.py --prefix /fastscratch/elhamod --name Phylo-VQVAE-transformer-phase7 --postfix 256img-phase7-uncoditional-phylo-to-nonphylo --base /home/elhamod/projects/taming-transformers/configs/phylo_vqgan_transformer-phase7-phylo-to-nonphylo.yaml -t True --gpus 0,

exit;





# Run these for the dataset you want before updating the custom_vqgan.yaml file and then running this script
# find /home/elhamod/data/Fish/test_padded_512 -name "*.???" > /home/elhamod/data/Fish/taming_transforms_fish_test_padded_512.txt
# find /home/elhamod/data/Fish/train_padded_512 -name "*.???" > /home/elhamod/data/Fish/taming_transforms_fish_train_padded_512.txt

# find /home/elhamod/data/Fish/test_padded_256 -name "*.???" > /home/elhamod/data/Fish/taming_transforms_fish_test_padded_256.txt
# find /home/elhamod/data/Fish/train_padded_256 -name "*.???" > /home/elhamod/data/Fish/taming_transforms_fish_train_padded_256.txt