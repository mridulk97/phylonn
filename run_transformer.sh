#!/bin/bash

#SBATCH --account=ml4science    #mabrownlab   #ml4science
#SBATCH --partition=a100_normal_q #a100_normal_q  #dgx_normal_q  #a100_normal_q
#SBATCH --time=0-8:00:00 
#SBATCH --gres=gpu:4
#SBATCH --nodes=1 --ntasks-per-node=1 --cpus-per-task=16
#SBATCH -o /fastscratch/elhamod/projects/phylonn/SLURM/slurm-%j.out

##########SBATCH -o ./SLURM/slurm-%j.out

# TODO: there is a bug. for some reason I need to reset again here.
module reset
module load Anaconda3/2020.11
source activate taming3 
module reset
source activate taming3 
which python

# python main.py --prefix /fastscratch/elhamod --name Phylo-VQVAE-transformer --postfix 256img-phase4-level3-finalstronger --base configs/phylo_vqgan_transformer-level3.yaml -t True --gpus 0, 
# python main.py --prefix /fastscratch/elhamod --name Phylo-VQVAE-transformer-phase6 --postfix 256img-phase6 --base /home/elhamod/projects/phylonn/configs/phylo_vqgan_transformer-phase6.yaml -t True --gpus 0, 
# python main.py --prefix /fastscratch/elhamod --name Phylo-VQVAE-transformer-phase6 --postfix 256img-phase6-less --base /home/elhamod/projects/phylonn/configs/phylo_vqgan_transformer-phase6-less.yaml -t True --gpus 0, 

# python main.py --prefix /fastscratch/elhamod --name Phylo-VQVAE-transformer-phase7 --postfix 256img-phase7-level3-idea3 --base /home/elhamod/projects/phylonn/configs/phylo_vqgan_transformer-phase7-level3-idea3.yaml -t True --gpus 0,
# python main.py --prefix /fastscratch/elhamod --name Phylo-VQVAE-transformer-phase7 --postfix 256img-phase7-level2-idea3 --base /home/elhamod/projects/phylonn/configs/phylo_vqgan_transformer-phase7-level2-idea3.yaml -t True --gpus 0,
# python main.py --prefix /fastscratch/elhamod --name Phylo-VQVAE-transformer-phase7 --postfix 256img-phase7-level1-idea3 --base /home/elhamod/projects/phylonn/configs/phylo_vqgan_transformer-phase7-level1-idea3.yaml -t True --gpus 0,
# python main.py --prefix /fastscratch/elhamod --name Phylo-VQVAE-transformer-phase7 --postfix 256img-phase7-level0-idea3 --base /home/elhamod/projects/phylonn/configs/phylo_vqgan_transformer-phase7-level0-idea3.yaml -t True --gpus 0,
# python main.py --prefix /fastscratch/elhamod --name Phylo-VQVAE-transformer-phase7 --postfix 256img-phase7-uncoditional-nonphylo --base /home/elhamod/projects/phylonn/configs/phylo_vqgan_transformer-phase7-uncoditional-nonphylo.yaml -t True --gpus 0,
# python main.py --prefix /fastscratch/elhamod --name Phylo-VQVAE-transformer-phase7 --postfix 256img-phase7-uncoditional --base /home/elhamod/projects/phylonn/configs/phylo_vqgan_transformer-phase7-uncoditional.yaml -t True --gpus 0,

# python main.py --prefix /fastscratch/elhamod --name Phylo-VQVAE-transformer-phase7 --postfix 256img-phase7-level3-partial_codes-morespace-samelayers-slower --base /home/elhamod/projects/phylonn/configs/phylo_vqgan_transformer-phase7-level3-morespace-samelayers-slower.yaml -t True --gpus 0,

# python main.py --prefix /fastscratch/elhamod --name Phylo-VQVAE-transformer-phase7 --postfix 256img-phase7-uncoditional-phylo-to-nonphylo --base /home/elhamod/projects/phylonn/configs/phylo_vqgan_transformer-phase7-phylo-to-nonphylo.yaml -t True --gpus 0,

# python main.py --prefix /fastscratch/elhamod --name Phylo-VQVAE-transformer-phase7 --postfix 256img-phase7-level3-morecodesperlevel --base /home/elhamod/projects/phylonn/configs/phylo_vqgan_transformer-phase7-level3-morecodesperlevel.yaml -t True --gpus 0,1,2

# python main.py --prefix /fastscratch/elhamod --name Phylo-VQVAE-transformer-phase7 --postfix 256img-phase7-level3-morenonphylo --base /home/elhamod/projects/phylonn/configs/phylo_vqgan_transformer-phase7-level3-morenonphylo.yaml -t True --gpus 0,1,2

# python main.py --prefix /fastscratch/elhamod --name Phylo-VQVAE-transformer-phase7 --postfix 256img-phase7-level3-morenonphylo-fewerlayers --base /home/elhamod/projects/phylonn/configs/phylo_vqgan_transformer-phase7-level3-morenonphylo-fewerlayers.yaml -t True --gpus 0,1,2

# python main.py --prefix /fastscratch/elhamod --name Phylo-VQVAE-transformer-phase7 --postfix 256img-phase7-level3-test --base /home/elhamod/projects/phylonn/configs/phylo_vqgan_transformer-phase7-level3-test.yaml -t True --gpus 0,1,2

# python main.py --prefix /fastscratch/elhamod --name Phylo-VQVAE-transformer-phase7 --postfix 256img-phase7-level3-lowerbeta-cyclical --base /home/elhamod/projects/phylonn/configs/phylo_vqgan_transformer-phase7-level3-lowerbeta.yaml -t True --gpus 0,1,2


# python main.py --prefix /fastscratch/elhamod --name Phylo-VQVAE-transformer-phase8 --postfix 256img-phase8-level3 --base /home/elhamod/projects/phylonn/configs/phylo_vqgan_transformer-phase8-level3.yaml -t True --gpus 0,
# python main.py --prefix /fastscratch/elhamod --name Phylo-VQVAE-transformer-phase8 --postfix 256img-phase8-level3-postfix --base /home/elhamod/projects/phylonn/configs/phylo_vqgan_transformer-phase8-level3-postfix.yaml -t True --gpus 0,
# python main.py --prefix /fastscratch/elhamod --name Phylo-VQVAE-transformer-phase8 --postfix 256img-phase8-level3-idea1 --base /home/elhamod/projects/phylonn/configs/phylo_vqgan_transformer-phase8-level3-idea1.yaml -t True --gpus 0,
# python main.py --prefix /fastscratch/elhamod --name Phylo-VQVAE-transformer-phase8 --postfix 256img-phase8-level3-onlyphylo --base /home/elhamod/projects/phylonn/configs/phylo_vqgan_transformer-phase8-level3-partialcodes.yaml -t True --gpus 0,

# python main.py --prefix /fastscratch/elhamod --name Phylo-VQVAE-transformer-phase8 --postfix 256img-phase8-level3-550epochs-4phylo-4nonphylo-idea1 --base /home/elhamod/projects/phylonn/configs/phylo_vqgan_transformer-phase8-level3-550epochs-4phylo-4nonphylo-idea1.yaml -t True --gpus 0,1,2
# python main.py --prefix /fastscratch/elhamod --name Phylo-VQVAE-transformer-phase8 --postfix 256img-phase8-level3-550epochs-4phylo-4nonphylo-morecodes-idea1 --base /home/elhamod/projects/phylonn/configs/phylo_vqgan_transformer-phase8-level3-550epochs-4phylo-4nonphylo-morecodes-idea1.yaml -t True --gpus 0,1,2

# python main.py --prefix /fastscratch/elhamod --name Phylo-VQVAE-transformer-phase8 --postfix 256img-phase8-level3-550epochs-4phylo-4nonphylo-idea5 --base /home/elhamod/projects/phylonn/configs/phylo_vqgan_transformer-phase8-level3-550epochs-4phylo-4nonphylo-idea5.yaml -t True --gpus 0,1
# python main.py --prefix /fastscratch/elhamod --name Phylo-VQVAE-transformer-phase8 --postfix 256img-phase8-level3-550epochs-4phylo-4nonphylo-idea5-lvl0 --base /home/elhamod/projects/phylonn/configs/phylo_vqgan_transformer-phase8-level3-550epochs-4phylo-4nonphylo-idea5-lvl0.yaml -t True --gpus 0,1
# python main.py --prefix /fastscratch/elhamod --name Phylo-VQVAE-transformer-phase8 --postfix 256img-phase8-level3-550epochs-4phylo-4nonphylo-idea5-lvl1 --base /home/elhamod/projects/phylonn/configs/phylo_vqgan_transformer-phase8-level3-550epochs-4phylo-4nonphylo-idea5-lvl1.yaml -t True --gpus 0,1
# python main.py --prefix /fastscratch/elhamod --name Phylo-VQVAE-transformer-phase8 --postfix 256img-phase8-level3-550epochs-4phylo-4nonphylo-idea5-lvl2 --base /home/elhamod/projects/phylonn/configs/phylo_vqgan_transformer-phase8-level3-550epochs-4phylo-4nonphylo-idea5-lvl2.yaml -t True --gpus 0,1

# python main.py --prefix /fastscratch/elhamod --name Phylo-VQVAE-transformer-phase8 --postfix 256img-phase8-level3-550epochs-4phylo-4nonphylo-idea5prefix --base /home/elhamod/projects/phylonn/configs/phylo_vqgan_transformer-phase8-level3-550epochs-4phylo-4nonphylo-idea5prefix.yaml -t True --gpus 0,1
# python main.py --prefix /fastscratch/elhamod --name Phylo-VQVAE-transformer-phase8 --postfix 256img-phase8-level3-550epochs-4phylo-4nonphylo-idea5prefix-lvl0 --base /home/elhamod/projects/phylonn/configs/phylo_vqgan_transformer-phase8-level3-550epochs-4phylo-4nonphylo-idea5prefix-lvl0.yaml -t True --gpus 0,1
# python main.py --prefix /fastscratch/elhamod --name Phylo-VQVAE-transformer-phase8 --postfix 256img-phase8-level3-550epochs-4phylo-4nonphylo-idea5prefix-lvl1 --base /home/elhamod/projects/phylonn/configs/phylo_vqgan_transformer-phase8-level3-550epochs-4phylo-4nonphylo-idea5prefix-lvl1.yaml -t True --gpus 0,
# python main.py --prefix /fastscratch/elhamod --name Phylo-VQVAE-transformer-phase8 --postfix 256img-phase8-level3-550epochs-4phylo-4nonphylo-idea5prefix-lvl2 --base /home/elhamod/projects/phylonn/configs/phylo_vqgan_transformer-phase8-level3-550epochs-4phylo-4nonphylo-idea5prefix-lvl2.yaml -t True --gpus 0,

# python main.py --prefix /fastscratch/elhamod --name Phylo-VQVAE-transformer-phase8 --postfix 256img-phase8-level3-testingoldtransformer --base /home/elhamod/projects/phylonn/configs/phylo_vqgan_transformer-phase8-level3-idea1-testingoldtransformer.yaml -t True --gpus 0,
# python main.py --prefix /fastscratch/elhamod --name Phylo-VQVAE-transformer-phase8 --postfix 256img-phase8-level3-testingnewtransformer --base /home/elhamod/projects/phylonn/configs/phylo_vqgan_transformer-phase8-level3-idea1-testingnewtransformer.yaml -t True --gpus 0,1

# python main.py --prefix /fastscratch/elhamod --name Phylo-VQVAE-transformer-phase8 --postfix 256img-phase8-level3-550epochs-4phylo-4nonphylo-idea1-newtransformer-test --base /home/elhamod/projects/phylonn/configs/phylo_vqgan_transformer-phase8-level3-550epochs-4phylo-4nonphylo-idea1-newtransformer.yaml -t True --gpus 0,
# python main.py --resume /fastscratch/elhamod/logs/2023-01-20T17-53-32_Phylo-VQVAE-transformer-phase8256img-phase8-level3-550epochs-4phylo-4nonphylo-idea1-newtransformer --prefix /fastscratch/elhamod --postfix 256img-phase8-level3-550epochs-4phylo-4nonphylo-idea1-newtransformer-test --base  /home/elhamod/projects/phylonn/configs/phylo_vqgan_transformer-phase8-level3-550epochs-4phylo-4nonphylo-idea1-newtransformer.yaml -t True --gpus 0,1,2

# python main.py --prefix /fastscratch/elhamod --name Phylo-VQVAE-transformer-phase8 --postfix 256img-phase8-level3-550epochs-4phylo-4nonphylo-idea1-newtransformer-t3 --base /home/elhamod/projects/phylonn/configs/phylo_vqgan_transformer-phase8-level3-550epochs-4phylo-4nonphylo-idea1-newtransformer-t3.yaml -t True --gpus 0,1,2

# python main.py --prefix /fastscratch/elhamod --name Phylo-VQVAE-transformer-phase8 --postfix 256img-phase8-originalVQGAN-smallbatch --base /home/elhamod/projects/phylonn/configs/original_VQGAN_transformer_phase8.yaml -t True --gpus 0,1,2

# python main.py --prefix /fastscratch/elhamod --name Phylo-VQVAE-transformer-phase8 --postfix 256img-phase8-conditional-idea5-nonphylo --base /home/elhamod/projects/phylonn/configs/phylo_vqgan_transformer-phase8-coditional-nonphylo.yaml -t True --gpus 0,1






# python main.py --prefix /fastscratch/elhamod --name Phylo-VQVAE-transformer-phase8 --postfix 256img-phase8-conditional-idea3-lvl3 --base /home/elhamod/projects/phylonn/configs/phylo_vqgan_transformer-phase8-level3-550epochs-4phylo-4nonphylo-idea3.yaml -t True --gpus 0,1

# python main.py --prefix /fastscratch/elhamod --postfix 256img-phase8-conditional-idea3-lvl0 --base /home/elhamod/projects/phylonn/configs/phylo_vqgan_transformer-phase8-level3-550epochs-4phylo-4nonphylo-idea3-lvl0.yaml -t True --gpus 0,1

# python main.py --resume /fastscratch/elhamod/logs/2023-01-25T00-19-52_Phylo-VQVAE-transformer-phase8256img-phase8-conditional-idea3-lvl1 --prefix /fastscratch/elhamod --postfix 256img-phase8-conditional-idea3-lvl1 --base /home/elhamod/projects/phylonn/configs/phylo_vqgan_transformer-phase8-level3-550epochs-4phylo-4nonphylo-idea3-lvl1.yaml -t True --gpus 0,1

# python main.py --resume /fastscratch/elhamod/logs/2023-01-25T00-19-52_Phylo-VQVAE-transformer-phase8256img-phase8-conditional-idea3-lv2 --prefix /fastscratch/elhamod --postfix 256img-phase8-conditional-idea3-lv2 --base /home/elhamod/projects/phylonn/configs/phylo_vqgan_transformer-phase8-level3-550epochs-4phylo-4nonphylo-idea3-lvl2.yaml -t True --gpus 0,1

# python main.py --prefix /fastscratch/elhamod --postfix 256img-phase8-conditional-idea3-nonphylo --base /home/elhamod/projects/phylonn/configs/phylo_vqgan_transformer-phase8-coditional-nonphylo.yaml -t True --gpus 0,1

# python main.py --prefix /fastscratch/elhamod --name Phylo-VQVAE-transformer-phase8 --postfix 256img-phase8-unconditional-idea3-nonphylo --base /home/elhamod/projects/phylonn/configs/phylo_vqgan_transformer-phase8-uncod-nonphylo.yaml -t True --gpus 0,1





# python main.py --name Phylo-VQVAE-transformer-phase8 --prefix /fastscratch/elhamod --postfix 256img-phase8-idea4-lvl2 --base /home/elhamod/projects/phylonn/configs/phylo_vqgan_transformer-phase8-idea4-4phylo-4nonphylo-lvl2.yaml -t True --gpus 0,1

# python main.py --name Phylo-VQVAE-transformer-phase8 --prefix /fastscratch/elhamod --postfix 256img-phase8-idea4-lvl1 --base /home/elhamod/projects/phylonn/configs/phylo_vqgan_transformer-phase8-idea4-4phylo-4nonphylo-lvl1.yaml -t True --gpus 0,1

# python main.py --prefix /fastscratch/elhamod --postfix 256img-phase8-idea4-lvl3 --name Phylo-VQVAE-transformer-phase8 --base /home/elhamod/projects/phylonn/configs/phylo_vqgan_transformer-phase8-idea4-4phylo-4nonphylo.yaml -t True --gpus 0,1

# python main.py --name Phylo-VQVAE-transformer-phase8 --prefix /fastscratch/elhamod --postfix 256img-phase8-idea4-nonphylo --base /home/elhamod/projects/phylonn/configs/phylo_vqgan_transformer-phase8-idea4-coditional-nonphylo.yaml -t True --gpus 0,1




# python main.py --prefix /fastscratch/elhamod --name Phylo-VQVAE-transformer-phase8 --postfix 256img-phase8-4phylo-4nonphylo-idea1-lvl3-fast --base configs/phylo_vqgan_transformer-phase8-level3-4phylo-4nonphylo-idea1-faster.yaml -t True --gpus 0,1



# python main.py --prefix /fastscratch/elhamod --name Phylo-VQVAE-transformer-phase8 --postfix 256img-phase8-4phylo-4nonphylo-idea1-lvl1-anti --base /home/elhamod/projects/phylonn/configs/phylo_vqgan_transformer-phase8-4phylo-4nonphylo-idea1-lvl1-anti.yaml -t True --gpus 0,1


# python main.py --prefix /fastscratch/elhamod --name Phylo-VQVAE-transformer-phase8 --postfix 256img-phase8-4phylo-4nonphylo-idea1-lvl2-anti --base /home/elhamod/projects/phylonn/configs/phylo_vqgan_transformer-phase8-4phylo-4nonphylo-idea1-lvl2-anti.yaml -t True --gpus 0,1


# python main.py --prefix /fastscratch/elhamod --name Phylo-VQVAE-transformer-phase8 --postfix 256img-phase8-4phylo-4nonphylo-idea1-lvl3-anti --base /home/elhamod/projects/phylonn/configs/phylo_vqgan_transformer-phase8-4phylo-4nonphylo-idea1-lvl3-anti.yaml -t True --gpus 0,1

# python main.py --prefix /fastscratch/elhamod --name Phylo-VQVAE-transformer-phase8 --postfix 256img-phase8-4phylo-4nonphylo-idea1-lvl3-sgd-faster --base /home/elhamod/projects/phylonn/configs/phylo_vqgan_transformer-phase8-level3-4phylo-4nonphylo-idea1-faster2-SGD.yaml-t True --gpus 0,1




# python main.py --prefix /fastscratch/elhamod --name Phylo-VQVAE-transformer-phase8 --postfix 256img-phase8-originalVQGAN-round2 --base /home/elhamod/projects/phylonn/configs/original_VQGAN_transformer_phase8-round2.yaml -t True --gpus 0,1,2,3

# python main.py --prefix /fastscratch/elhamod --name Phylo-VQVAE-transformer-phase8 --postfix 256img-phase8-originalVQGAN-round2-morelayers --base /home/elhamod/projects/phylonn/configs/original_VQGAN_transformer_phase8-round2-morelayers.yaml -t True --gpus 0,1,2,3

# python main.py --prefix /fastscratch/elhamod --name Phylo-VQVAE-transformer-phase8 --postfix 256img-phase8-originalVQGAN-round2-smallerbatch --base /home/elhamod/projects/phylonn/configs/original_VQGAN_transformer_phase8-round2-smallerbatch.yaml -t True --gpus 0,1,2,3


# python main.py --prefix /fastscratch/elhamod --name Phylo-VQVAE-transformer-phase8 --postfix 256img-phase8-4phylo-2nonphylo-idea1-lvl1 --base /home/elhamod/projects/phylonn/configs/phylo_vqgan_transformer-phase8-4phylo-4nonphylo-idea1-lvl1.yaml -t True --gpus 0,1

# python main.py --prefix /fastscratch/elhamod --name Phylo-VQVAE-transformer-phase8 --postfix 256img-phase8-4phylo-2nonphylo-idea1-lvl2 --base /home/elhamod/projects/phylonn/configs/phylo_vqgan_transformer-phase8-4phylo-4nonphylo-idea1-lvl2.yaml -t True --gpus 0,1

python main.py --prefix /fastscratch/elhamod --name Phylo-VQVAE-transformer-phase8 --postfix 256img-phase8-4phylo-2nonphylo-idea1-lvl0 --base /home/elhamod/projects/phylonn/configs/phylo_vqgan_transformer-phase8-4phylo-4nonphylo-idea1-lvl0.yaml -t True --gpus 0,1

# python main.py --prefix /fastscratch/elhamod --name Phylo-VQVAE-transformer-phase8 --postfix 256img-phase8-4phylo-2nonphylo-idea1-lvl3 --base /home/elhamod/projects/phylonn/configs/phylo_vqgan_transformer-phase8-4phylo-4nonphylo-idea1-lvl3.yaml -t True --gpus 0,1,3,4


# bash /home/elhamod/projects/phylonn/scripts/make_samples_phylo.sh
exit;





# Run these for the dataset you want before updating the custom_vqgan.yaml file and then running this script
# find /home/elhamod/data/Fish/test_padded_512 -name "*.???" > /home/elhamod/data/Fish/fish_test_padded_512.txt
# find /home/elhamod/data/Fish/train_padded_512 -name "*.???" > /home/elhamod/data/Fish/fish_train_padded_512.txt

# find /home/elhamod/data/Fish/test_padded_256 -name "*.???" > /home/elhamod/data/Fish/fish_test_padded_256.txt
# find /home/elhamod/data/Fish/train_padded_256 -name "*.???" > /home/elhamod/data/Fish/fish_train_padded_256.txt


# find /fastscratch/elhamod/logs/2023-02-01T13-45-34_Phylo-VQVAE-transformer-phase8256img-phase8-originalVQGAN-round2-smallerbatch/transformer-generated/008190_100_1.0/samples -name "*.???" > /fastscratch/elhamod/logs/2023-02-01T13-45-34_Phylo-VQVAE-transformer-phase8256img-phase8-originalVQGAN-round2-smallerbatch/transformer-generated/008190_100_1.0/fish_test_padded_512.txt

# find /fastscratch/elhamod/logs/2023-01-28T17-44-33_Phylo-VQVAE-transformer-phase8256img-phase8-4phylo-4nonphylo-idea1-lvl3/figs/transformer_generated_dataset/species_transfromer_generated -name "*.???" > /fastscratch/elhamod/logs/2023-01-28T17-44-33_Phylo-VQVAE-transformer-phase8256img-phase8-4phylo-4nonphylo-idea1-lvl3/figs/transformer_generated_dataset/fish_test_padded_512.txt