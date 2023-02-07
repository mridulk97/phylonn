#!/bin/bash

#SBATCH --account=ml4science
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

python analysis/test_resnet.py 
# python analysis/test_resnet.py --config analysis/configs/test_resnet_level0.yaml
# python analysis/test_resnet.py --config analysis/configs/test_resnet_level1.yaml
# python analysis/test_resnet.py --config analysis/configs/test_resnet_level2.yaml

python analysis/test_resnet.py --config analysis/configs/test_resnet-train.yaml
# python analysis/test_resnet.py --config analysis/configs/test_resnet_level0-train.yaml
# python analysis/test_resnet.py --config analysis/configs/test_resnet_level1-train.yaml
# python analysis/test_resnet.py --config analysis/configs/test_resnet_level2-train.yaml

python analysis/test_resnet.py --config analysis/configs/test_resnet-val.yaml
# python analysis/test_resnet.py --config analysis/configs/test_resnet_level0-val.yaml
# python analysis/test_resnet.py --config analysis/configs/test_resnet_level1-val.yaml
# python analysis/test_resnet.py --config analysis/configs/test_resnet_level2-val.yaml

# python analysis/test_resnet.py --config /home/elhamod/projects/taming-transformers/analysis/configs/test_resnet_originalvqgan.yaml
# python analysis/test_resnet.py --config /home/elhamod/projects/taming-transformers/analysis/configs/test_resnet_originalvqgan-level0.yaml
# python analysis/test_resnet.py --config /home/elhamod/projects/taming-transformers/analysis/configs/test_resnet_originalvqgan-level1.yaml
# python analysis/test_resnet.py --config /home/elhamod/projects/taming-transformers/analysis/configs/test_resnet_originalvqgan-level2.yaml



exit;