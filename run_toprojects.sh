#!/bin/bash

#SBATCH --account=ml4science
#SBATCH --partition=dgx_normal_q
#SBATCH --time=2-00:00:00 
#SBATCH --gres=gpu:1
#SBATCH --nodes=1 --ntasks-per-node=1 --cpus-per-task=8
#SBATCH -o /fastscratch/elhamod/projects/phylonn/SLURM/slurm-%j.out
##########SBATCH -o ./SLURM/slurm-%j.out

# TODO: there is a bug. for some reason I need to reset again here.
module reset
module load Anaconda3/2020.11
source activate taming3 
module reset
source activate taming3 
which python


# mkdir -p /projects/ml4science/phylonn/elhamod/data/Fish
# cp -avr /home/elhamod/data/Fish/fish_test_padded_256.txt /projects/ml4science/phylonn/elhamod/data/Fish/fish_test_padded_256.txt
# cp -avr /home/elhamod/data/Fish/fish_train_padded_256.txt /projects/ml4science/phylonn/elhamod/data/Fish/fish_train_padded_256.txt
# cp -avr /home/elhamod/data/Fish/fish_test_padded_512.txt /projects/ml4science/phylonn/elhamod/data/Fish/fish_test_padded_512.txt
# cp -avr /home/elhamod/data/Fish/fish_train_padded_512.txt /projects/ml4science/phylonn/elhamod/data/Fish/fish_train_padded_512.txt
# cp -avr /home/elhamod/data/Fish/cleaned_metadata.tre /projects/ml4science/phylonn/elhamod/data/Fish/cleaned_metadata.tre
# cp -avr /home/elhamod/data/Fish/name_conversion.pkl /projects/ml4science/phylonn/elhamod/data/Fish/name_conversion.pkl
# cp -avr /home/elhamod/data/Fish/test_padded_256 /projects/ml4science/phylonn/elhamod/data/Fish/test_padded_256
# cp -avr /home/elhamod/data/Fish/train_padded_256 /projects/ml4science/phylonn/elhamod/data/Fish/train_padded_256
# cp -avr /home/elhamod/data/Fish/test_padded_512 /projects/ml4science/phylonn/elhamod/data/Fish/test_padded_512
# cp -avr /home/elhamod/data/Fish/train_padded_512 /projects/ml4science/phylonn/elhamod/data/Fish/train_padded_512

# cp -avr /home/elhamod/data/Fish/fish_test_padded_256_out.txt /projects/ml4science/phylonn/elhamod/data/Fish/fish_test_padded_256_out.txt
# cp -avr /home/elhamod/data/Fish/fish_train_padded_256_out.txt /projects/ml4science/phylonn/elhamod/data/Fish/fish_train_padded_256_out.txt


# mkdir -p /projects/ml4science/phylonn/elhamod/projects/phylonn/logs/512pixels_512embedding
# mkdir -p /projects/ml4science/phylonn/elhamod/projects/phylonn/logs/512pixels_256embedding
# mkdir -p /projects/ml4science/phylonn/elhamod/projects/phylonn/logs/256pixels_256embedding
# mkdir -p  /projects/ml4science/phylonn/elhamod/projects/phylonn/SLURM
# cp -avr /home/elhamod/projects/phylonn/logs/256pixels_256embedding/checkpoints /projects/ml4science/phylonn/elhamod/projects/phylonn/logs/256pixels_256embedding/checkpoints
# cp -avr /home/elhamod/projects/phylonn/logs/512pixels_256embedding/checkpoints /projects/ml4science/phylonn/elhamod/projects/phylonn/logs/512pixels_256embedding/checkpoints
# cp -avr /home/elhamod/projects/phylonn/logs/512pixels_512embedding/checkpoints /projects/ml4science/phylonn/elhamod/projects/phylonn/logs/512pixels_512embedding/checkpoints
# cp -avr /home/elhamod/projects/phylonn/logs/256pixels_256embedding/configs /projects/ml4science/phylonn/elhamod/projects/phylonn/logs/256pixels_256embedding/configs
# cp -avr /home/elhamod/projects/phylonn/logs/512pixels_256embedding/configs /projects/ml4science/phylonn/elhamod/projects/phylonn/logs/512pixels_256embedding/configs
# cp -avr /home/elhamod/projects/phylonn/logs/512pixels_512embedding/configs /projects/ml4science/phylonn/elhamod/projects/phylonn/logs/512pixels_512embedding/configs


cp -avr /fastscratch/elhamod /projects/ml4science/phylonn


exit;
