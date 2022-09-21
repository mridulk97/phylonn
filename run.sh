#!/bin/bash

#SBATCH --account=ml4science
#SBATCH --partition=a100_normal_q
#SBATCH --time=1-12:00:00 
#SBATCH --gres=gpu:2 
#SBATCH --nodes=1 --ntasks-per-node=1 --cpus-per-task=8


echo start load env and run python

module reset

module load Anaconda3/2020.11
module load gcc/8.2.0

source activate taming3 

python main.py --base configs/custom_vqgan.yaml -t True --gpus 0,1

exit;





# find /home/elhamod/data/Fish/test -name "*.???" > /home/elhamod/data/Fish/taming_transforms_fish_test.txt
# find /home/elhamod/data/Fish/train -name "*.???" > /home/elhamod/data/Fish/taming_transforms_fish_train.txt