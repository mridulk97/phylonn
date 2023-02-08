#!/bin/bash

#SBATCH --account=imageomics-biosci  #mabrownlab   #ml4science
#SBATCH --partition=dgx_normal_q
#SBATCH --time=0-05:00:00 
#SBATCH --gres=gpu:8
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

# python main.py --prefix /fastscratch/elhamod --name Phylo-VQVAE-phase6 --postfix 256img-phase6 --base /home/elhamod/projects/phylonn/configs/custom_vqgan-256emb-256img-phylo-vqvae-phase6.yaml -t True --gpus 0,1,2
# python main.py --prefix /fastscratch/elhamod --name Phylo-VQVAE-phase6 --postfix 256img-phase6-phyloalone --base /home/elhamod/projects/phylonn/configs/custom_vqgan-256emb-256img-phylo-vqvae-phase6-phyloalone.yaml -t True --gpus 0,1,2
# python main.py --prefix /fastscratch/elhamod --name Phylo-VQVAE-phase6 --postfix 256img-phase6-speciesout --base /home/elhamod/projects/phylonn/configs/custom_vqgan-256emb-256img-phylo-vqvae-phase6-speciesout.yaml -t True --gpus 0,1,2
# python main.py --prefix /fastscratch/elhamod --name Phylo-VQVAE-phase6 --postfix 256img-phase6-phylohigh --base /home/elhamod/projects/phylonn/configs/custom_vqgan-256emb-256img-phylo-vqvae-phase6-phylohigh.yaml -t True --gpus 0,1,2
# python main.py --prefix /fastscratch/elhamod --name Phylo-VQVAE-phase6 --postfix 256img-phase6-morech --base /home/elhamod/projects/phylonn/configs/custom_vqgan-256emb-256img-phylo-vqvae-phase6-morech.yaml -t True --gpus 0,1,2
# python main.py --prefix /fastscratch/elhamod --name Phylo-VQVAE-phase6 --postfix 256img-phase6-phyloquantize --base /home/elhamod/projects/phylonn/configs/custom_vqgan-256emb-256img-phylo-vqvae-phase6-phyloplusquant.yaml -t True --gpus 0,1,2
# python main.py --prefix /fastscratch/elhamod --name Phylo-VQVAE-phase6 --postfix 256img-phase6-phyloquantizerec --base /home/elhamod/projects/phylonn/configs/custom_vqgan-256emb-256img-phylo-vqvae-phase6-phyloplusquantrec.yaml -t True --gpus 0,1,2
# python main.py --prefix /fastscratch/elhamod --name Phylo-VQVAE-phase6 --postfix 256img-phase6-noorth --base /home/elhamod/projects/phylonn/configs/custom_vqgan-256emb-256img-phylo-vqvae-phase6-noorth.yaml -t True --gpus 0,1,2
# python main.py --prefix /fastscratch/elhamod --name Phylo-VQVAE-phase6 --postfix 256img-phase6-highortho --base /home/elhamod/projects/phylonn/configs/custom_vqgan-256emb-256img-phylo-vqvae-phase6-highortho.yaml -t True --gpus 0,1,2
# python main.py --prefix /fastscratch/elhamod --name Phylo-VQVAE-phase6 --postfix 256img-phase6-lowerlr --base /home/elhamod/projects/phylonn/configs/custom_vqgan-256emb-256img-phylo-vqvae-phase6-lowerlr.yaml -t True --gpus 0,1,2
# python main.py --prefix /fastscratch/elhamod --name Phylo-VQVAE-phase6 --postfix 256img-phase6-fixoptimization --base /home/elhamod/projects/phylonn/configs/custom_vqgan-256emb-256img-phylo-vqvae-phase6-autooptimization.yaml -t True --gpus 0,1,2
# python main.py --prefix /fastscratch/elhamod --name Phylo-VQVAE-phase6 --postfix 256img-phase6-higherlr --base /home/elhamod/projects/phylonn/configs/custom_vqgan-256emb-256img-phylo-vqvae-phase6-higherlr.yaml -t True --gpus 0,1,2

# python main.py --prefix /fastscratch/elhamod --name Phylo-VQVAE-phase7 --postfix 256img-mediumbeta --base /home/elhamod/projects/phylonn/configs/archived/custom_vqgan-256emb-256img-phylo-vqvae-phase7-lessbeta.yaml -t True --gpus 0,1,2

# python main.py --prefix /fastscratch/elhamod --name Phylo-VQVAE-phase7 --postfix 256img-largeremebdding --base /home/elhamod/projects/phylonn/configs/archived/custom_vqgan-256emb-256img-phylo-vqvae-phase7-largercode.yaml -t True --gpus 0,1,2

# python main.py --prefix /fastscratch/elhamod --name Phylo-VQVAE-phase7 --postfix 256img-morelayers --base /home/elhamod/projects/phylonn/configs/archived/custom_vqgan-256emb-256img-phylo-vqvae-phase7-morelayers.yaml -t True --gpus 0,1,2
# python main.py --prefix /fastscratch/elhamod --name Phylo-VQVAE-phase7 --postfix 256img-morephyloweight --base /home/elhamod/projects/phylonn/configs/archived/custom_vqgan-256emb-256img-phylo-vqvae-phase7-morephyloweight.yaml -t True --gpus 0,1,2
# python main.py --prefix /fastscratch/elhamod --name Phylo-VQVAE-phase7 --postfix 256img-noanti --base /home/elhamod/projects/phylonn/configs/archived/custom_vqgan-256emb-256img-phylo-vqvae-phase7-noanti.yaml -t True --gpus 0,1,2
# python main.py --prefix /fastscratch/elhamod --name Phylo-VQVAE-phase7 --postfix 256img-nokernel --base /home/elhamod/projects/phylonn/configs/archived/custom_vqgan-256emb-256img-phylo-vqvae-phase7-nokernelorth.yaml -t True --gpus 0,1,2


# python main.py --prefix /fastscratch/elhamod --name Phylo-VQVAE-phase7 --postfix 256img-morenonphylo --base /home/elhamod/projects/phylonn/configs/custom_vqgan-256emb-256img-phylo-vqvae-phase7-lessbeta-morenonphylo.yaml -t True --gpus 0,1,2
# python main.py --prefix /fastscratch/elhamod --name Phylo-VQVAE-phase7 --postfix 256img-morecodesperlevel --base /home/elhamod/projects/phylonn/configs/custom_vqgan-256emb-256img-phylo-vqvae-phase7-lessbeta-morecodesperlevel.yaml -t True --gpus 0,1,2
# python main.py --prefix /fastscratch/elhamod --name Phylo-VQVAE-phase7 --postfix 256img-largercodebook --base /home/elhamod/projects/phylonn/configs/custom_vqgan-256emb-256img-phylo-vqvae-phase7-lessbeta-largercodebook.yaml -t True --gpus 0,1,2

# python main.py --prefix /fastscratch/elhamod --name Phylo-VQVAE-phase7 --postfix 256img-morech --base /home/elhamod/projects/phylonn/configs/custom_vqgan-256emb-256img-phylo-vqvae-phase7-morech.yaml -t True --gpus 0,1,2

# python main.py --prefix /fastscratch/elhamod --name Phylo-VQVAE-phase7 --postfix 256img-morerecweight --base /home/elhamod/projects/phylonn/configs/custom_vqgan-256emb-256img-phylo-vqvae-phase7-lessbeta-morerecweight.yaml -t True --gpus 0,1,2

# python main.py --resume /fastscratch/elhamod/logs/2022-12-26T17-56-10_Phylo-VQVAE-phase7256img-veryslow --prefix /fastscratch/elhamod --postfix 256img-veryslow --base /fastscratch/elhamod/logs/2022-12-26T17-56-10_Phylo-VQVAE-phase7256img-veryslow/configs/2022-12-26T17-56-10-project.yaml -t True --gpus 0,1,2

# python main.py --prefix /fastscratch/elhamod --name Phylo-VQVAE-phase7 --postfix 256img-addbaserecloss --base /home/elhamod/projects/phylonn/configs/custom_vqgan-256emb-256img-phylo-vqvae-phase7-lessbeta-addbaseloss.yaml -t True --gpus 0,1,2
# python main.py --resume /home/elhamod/projects/phylonn/logs/2022-10-14T10-40-42_Phylo-VQVAE256img-afterhyperp-combined --postfix 256img-afterhyperp-combined --base configs/custom_vqgan-256emb-512img-phylo-vqvae-phyloloss-afterhyperp-combination-nopassthrough.yaml -t True --gpus 0,

# python main.py --prefix /fastscratch/elhamod --name Phylo-VQVAE-phase7 --postfix 256img-lowerbeta-classificationnorm --base /home/elhamod/projects/phylonn/configs/custom_vqgan-256emb-256img-phylo-vqvae-phase7-lessbeta-normalizeclassification.yaml -t True --gpus 0,1,2


# python main.py --prefix /fastscratch/elhamod --name Phylo-VQVAE-phase7 --postfix 256img-lowerbeta-higheranti --base /home/elhamod/projects/phylonn/configs/custom_vqgan-256emb-256img-phylo-vqvae-phase7-lessbeta-higheranti.yaml -t True --gpus 0,1,2

# python main.py --prefix /fastscratch/elhamod --name Phylo-VQVAE-phase7 --postfix 256img-lowerbeta-binarycodes --base /home/elhamod/projects/phylonn/configs/custom_vqgan-256emb-256img-phylo-vqvae-phase7-lessbeta-binarycodes.yaml -t True --gpus 0,1,2

# python main.py --prefix /fastscratch/elhamod --name Phylo-VQVAE-phase7 --postfix 256img-lowerbeta-higherbaseloss --base /home/elhamod/projects/phylonn/configs/custom_vqgan-256emb-256img-phylo-vqvae-phase7-lessbeta-higherbaseloss.yaml -t True --gpus 0,1,2

# python main.py --prefix /fastscratch/elhamod --name Phylo-VQVAE-phase7 --postfix 256img-lowerbeta-labelsmoothing --base /home/elhamod/projects/phylonn/configs/custom_vqgan-256emb-256img-phylo-vqvae-phase7-lessbeta-labelsmoothing.yaml -t True --gpus 0,1,2


# python main.py --prefix /fastscratch/elhamod --name Phylo-VQVAE-phase7 --postfix 256img-lessbeta-lrfactor --base /home/elhamod/projects/phylonn/configs/custom_vqgan-256emb-256img-phylo-vqvae-phase7-lessbeta-lrfactor.yaml -t True --gpus 0,1,2
# python main.py --prefix /fastscratch/elhamod --name Phylo-VQVAE-phase7 --postfix 256img-lessnonphylo --base /home/elhamod/projects/phylonn/configs/custom_vqgan-256emb-256img-phylo-vqvae-phase7-lessbeta-lessnonphylo.yaml -t True --gpus 0,1,2

# python main.py --prefix /fastscratch/elhamod --name Phylo-VQVAE-phase7 --postfix 256img-morenonphyloanti --base /home/elhamod/projects/phylonn/configs/custom_vqgan-256emb-256img-phylo-vqvae-phase7-lessbeta-morenonphyloanti.yaml -t True --gpus 0,1,2

# python main.py --prefix /fastscratch/elhamod --name Phylo-VQVAE-phase7 --postfix 256img-originalvqgan-imagenetbg --base /home/elhamod/projects/phylonn/configs/custom_vqgan-256-meanbackground.yaml -t True --gpus 0,1,2
# python main.py --prefix /fastscratch/elhamod --name Phylo-VQVAE-phase7 --postfix 256img-originalvqgan-imagenetbg-aug --base /home/elhamod/projects/phylonn/configs/custom_vqgan-256-meanbackground-augmentation.yaml -t True --gpus 0,1,2

# python main.py --prefix /fastscratch/elhamod --name Phylo-VQVAE-phase7 --postfix 256img-originalvqgan-retry --base /home/elhamod/projects/phylonn/configs/custom_vqgan-256-original-retry.yaml -t True --gpus 0,1

# python main.py --prefix /fastscratch/elhamod --name Phylo-VQVAE-phase7 --postfix 256img-lowerbeta-augimagenetbg --base /home/elhamod/projects/phylonn/configs/custom_vqgan-256emb-256img-phylo-vqvae-phase7-lessbeta-augimagenetbg.yaml -t True --gpus 0,1,2

# python main.py --prefix /fastscratch/elhamod --name Phylo-VQVAE-phase7 --postfix 256img-lowerbeta-antiv2 --base /home/elhamod/projects/phylonn/configs/custom_vqgan-256emb-256img-phylo-vqvae-phase7-lessbeta-antiv2.yaml -t True --gpus 0,1,2


# python main.py --prefix /fastscratch/elhamod --name Phylo-VQVAE-phase7 --postfix 256img-lowerbeta-antiv3 --base /home/elhamod/projects/phylonn/configs/custom_vqgan-256emb-256img-phylo-vqvae-phase7-lessbeta-antiv3.yaml -t True --gpus 0,1,2

# python main.py --prefix /fastscratch/elhamod --name Phylo-VQVAE-phase7 --postfix 256img-lowerbeta-antiv3-fixed --base /home/elhamod/projects/phylonn/configs/custom_vqgan-256emb-256img-phylo-vqvae-phase7-lessbeta-antiv3-fixed.yaml -t True --gpus 0,1,2


# python main.py --prefix /fastscratch/elhamod --name Phylo-VQVAE-phase8 --postfix 256img-smallerbatch --base /home/elhamod/projects/phylonn/configs/custom_vqgan-256emb-256img-phylo-vqvae-phase8-smallerbatch.yaml -t True --gpus 0,1,2
# python main.py --prefix /fastscratch/elhamod --name Phylo-VQVAE-phase8 --postfix 256img --base /home/elhamod/projects/phylonn/configs/custom_vqgan-256emb-256img-phylo-vqvae-phase8.yaml -t True --gpus 0,1,2
# python main.py --prefix /fastscratch/elhamod --name Phylo-VQVAE-phase8 --postfix 256img-withaugmentation --base /home/elhamod/projects/phylonn/configs/custom_vqgan-256emb-256img-phylo-vqvae-phase8-withaugmentation.yaml -t True --gpus 0,1,2

# python main.py --prefix /fastscratch/elhamod --name Phylo-VQVAE-phase8 --postfix 256img-4phylo-4nonphylo --base /home/elhamod/projects/phylonn/configs/custom_vqgan-256emb-256img-phylo-vqvae-phase8-4phylo-4nonphylo.yaml -t True --gpus 0,1,2
# python main.py --resume /fastscratch/elhamod/logs/2023-01-13T15-13-08_Phylo-VQVAE-phase8256img-4phylo-4nonphylo --prefix /fastscratch/elhamod --postfix 256img-4phylo-4nonphylo --base /home/elhamod/projects/phylonn/configs/custom_vqgan-256emb-256img-phylo-vqvae-phase8-4phylo-4nonphylo.yaml -t True --gpus 0,1

# python main.py --prefix /fastscratch/elhamod --name Phylo-VQVAE-phase8 --postfix 256img-morecodeslevel --base /home/elhamod/projects/phylonn/configs/custom_vqgan-256emb-256img-phylo-vqvae-phase8-4phylo-4nonphylo-morecodeslevel.yaml -t True --gpus 0,1,2

# python main.py --prefix /fastscratch/elhamod --name Phylo-VQVAE-phase8 --postfix 256img-4phylo-4nonphylo-selfattention --base /home/elhamod/projects/phylonn/configs/custom_vqgan-256emb-256img-phylo-vqvae-phase8-4phylo-4nonphylo-selfattention.yaml -t True --gpus 0,1,2

# python main.py --prefix /fastscratch/elhamod --name Phylo-VQVAE-phase8 --postfix 256img-4phylo-4nonphylo-CUB --base /home/elhamod/projects/phylonn/configs/custom_vqgan-256emb-256img-phylo-vqvae-phase8-4phylo-4nonphylo-CUB.yaml -t True --gpus 0,1,2

# python main.py --resume /fastscratch/elhamod/logs/2023-01-28T01-01-57_Phylo-VQVAE-phase8256img-4phylo-4nonphylo-CUB --prefix /fastscratch/elhamod --postfix 256img-4phylo-4nonphylo-CUB --base /home/elhamod/projects/phylonn/configs/custom_vqgan-256emb-256img-phylo-vqvae-phase8-4phylo-4nonphylo-CUB.yaml -t True --gpus 0,1,2 #,3,4,5


# python main.py --prefix /fastscratch/elhamod --name Phylo-VQVAE-phase8 --postfix 256img-4phylo-4nonphylo-CUB-faster --base /home/elhamod/projects/phylonn/configs/custom_vqgan-256emb-256img-phylo-vqvae-phase8-4phylo-4nonphylo-CUB-faster.yaml -t True --gpus 0,1,2

# python main.py --prefix /fastscratch/elhamod --name Phylo-VQVAE-phase8 --postfix 256img-4phylo-4nonphylo-CUB-round2 --base /home/elhamod/projects/phylonn/configs/custom_vqgan-256emb-256img-phylo-vqvae-phase8-4phylo-4nonphylo-CUB-round2.yaml -t True --gpus 0,1,2





python main.py --prefix /fastscratch/elhamod --name Phylo-VQVAE-phase8 --postfix 256img-4phylo-4nonphylo-CUB-round3 --base /home/elhamod/projects/phylonn/configs/custom_vqgan-256emb-256img-phylo-vqvae-phase8-4phylo-4nonphylo-CUB-round3.yaml -t True --gpus 0,1,2,3,4,5,6,7

# python main.py --prefix /fastscratch/elhamod --name Phylo-VQVAE-phase8 --postfix 256img-4phylo-4nonphylo-fewercodes --base /home/elhamod/projects/phylonn/configs/custom_vqgan-256emb-256img-phylo-vqvae-phase8-4phylo-4nonphylo-fewercodes.yaml -t True --gpus 0,1,2,3,4

# python main.py --prefix /fastscratch/elhamod --name Phylo-VQVAE-phase8 --postfix 256img-4phylo-4nonphylo-fewernonphylodim --base /home/elhamod/projects/phylonn/configs/custom_vqgan-256emb-256img-phylo-vqvae-phase8-4phylo-4nonphylo-fewernonphylodim.yaml -t True --gpus 0,1,2,3,4

# python main.py --resume /fastscratch/elhamod/logs/2023-01-31T22-56-46_Phylo-VQVAE-phase8256img-4phylo-4nonphylo-fewernonphylodim-speciesout --prefix /fastscratch/elhamod --postfix 256img-4phylo-4nonphylo-fewernonphylodim-speciesout --base /home/elhamod/projects/phylonn/configs/custom_vqgan-256emb-256img-phylo-vqvae-phase8-4phylo-4nonphylo-fewernonphylodim-speciesout.yaml -t True --gpus 0,1,2,3,4

# python main.py --name Phylo-VQVAE-phase8 --prefix /fastscratch/elhamod /fastscratch/elhamod/logs/2023-01-31T22-56-46_Phylo-VQVAE-phase8256img-4phylo-4nonphylo-fewernonphylodim-withdisc --postfix 256img-4phylo-4nonphylo-fewernonphylodim-speciesout --base /home/elhamod/projects/phylonn/configs/custom_vqgan-256emb-256img-phylo-vqvae-phase8-4phylo-4nonphylo-fewernonphylodim-withdisc.yaml -t True --gpus 0,1,2,3,4




exit;





# Run these for the dataset you want before updating the custom_vqgan.yaml file and then running this script
# find /home/elhamod/data/Fish/test_padded_512 -name "*.???" > /home/elhamod/data/Fish/fish_test_padded_512.txt
# find /home/elhamod/data/Fish/train_padded_512 -name "*.???" > /home/elhamod/data/Fish/fish_train_padded_512.txt

# find /home/elhamod/data/Fish/test_padded_256 -name "*.???" > /home/elhamod/data/Fish/fish_test_padded_256.txt
# find /home/elhamod/data/Fish/train_padded_256 -name "*.???" > /home/elhamod/data/Fish/fish_train_padded_256.txt