#!/bin/bash

#SBATCH --account=ml4science
#SBATCH --partition=a100_normal_q
#SBATCH --time=2-00:00:00 
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

# python main.py --name Phylo-VQVAE --base configs/custom_vqgan-256emb-512img-phylo-vqvae-afterhyperp.yaml -t True --gpus 0, #1 # crashes... not enough memory?!
# python main.py --name Phylo-VQVAE --base configs/custom_vqgan-256emb-512img-phylo-vqvae.yaml -t True --gpus 0, #1
# python main.py --name Phylo-VQVAE --base configs/custom_vqgan-256emb-512img-phylo-vqvae-phyloloss.yaml -t True --gpus 0,
# python main.py --name Phylo-VQVAE --postfix 512img-afterhyperp --base configs/custom_vqgan-256emb-512img-phylo-vqvae-phyloloss-afterhyperp.yaml -t True --gpus 0,

# python main.py --name Phylo-VQVAE --postfix 512img-afterhyperp --base configs/custom_vqgan-512emb-512img-phylo-vqvae-phyloloss-afterhyperp-ch64.yaml -t True --gpus 0,
# python main.py --name Phylo-VQVAE --postfix 512img-afterhyperp --base configs/custom_vqgan-512emb-512img-phylo-vqvae-phyloloss-afterhyperp-combination-nopassthrough.yaml -t True --gpus 0,
# python main.py --name Phylo-VQVAE --postfix 512img-afterhyperp-no_orthogonality --base configs/custom_vqgan-512emb-512img-phylo-vqvae-phyloloss-afterhyperp-no_orthogonality.yaml -t True --gpus 0,

# resume
# python main.py --resume /home/elhamod/projects/taming-transformers/logs/2022-10-14T11-39-39_Phylo-VQVAE512img-afterhyperp --postfix 512img-afterhyperp --base configs/custom_vqgan-512emb-512img-phylo-vqvae-phyloloss-afterhyperp-combination-nopassthrough.yaml -t True --gpus 0,

# python main.py --name Phylo-VQVAE --postfix 256img-afterhyperp-combined-anticlass --base configs/custom_vqgan-256emb-512img-phylo-vqvae-phyloloss-afterhyperp-anticlassloss.yaml -t True --gpus 0,
# python main.py --name Phylo-VQVAE --postfix 512img-afterhyperp-combined-anticlass --base configs/custom_vqgan-512img-phylo-vqvae-phyloloss-afterhyperp-anticlassloss.yaml -t True --gpus 0,
# python main.py --name Phylo-VQVAE --postfix 512img-afterhyperp-combined-4cbperlevel --base configs/custom_vqgan-512img-phylo-vqvae-phyloloss-afterhyperp-combination-nopassthrough-4cbperlevel.yaml -t True --gpus 0,
# python main.py --name Phylo-VQVAE --postfix 512img-phylo-vqvae-phase2-morech-8cbperlevel --base configs/custom_vqgan-256emb-512img-phylo-vqvae-phase2-morech-8cbperlevel.yaml -t True --gpus 0,
# python main.py --name Phylo-VQVAE --postfix 256img-phylo-vqvae-phase2-morech-8cbperlevel --base configs/custom_vqgan-256emb-256img-phylo-vqvae-phase2-morech-8cbperlevel.yaml -t True --gpus 0,

# python main.py --prefix /fastscratch/elhamod --name Phylo-VQVAE --postfix 512img-phylo-vqvae-phase2-morech-8cbperlevel-morenonattr --base configs/custom_vqgan-256emb-256img-phylo-vqvae-phase2-morech-8cbperlevel-morenonattr.yaml -t True --gpus 0,
# python main.py --name Phylo-VQVAE --postfix 512img-phylo-vqvae-phase2-morech-8cbperlevel-morenonattr-multiclass --base configs/custom_vqgan-256emb-256img-phylo-vqvae-phase2-morech-8cbperlevel-morenonattr-multiclass.yaml -t True --gpus 0,


# python main.py --name Phylo-VQVAE --postfix 256img-phylo-vqvae-phase2-disentanglerconv --base configs/custom_vqgan-256emb-256img-phylo-vqvae-phase3-disentanglerconv.yaml -t True --gpus 0,
# python main.py --name Phylo-VQVAE --postfix 256img-phylo-vqvae-phase2-disentanglerconv-morephylocodes --base configs/custom_vqgan-256emb-256img-phylo-vqvae-phase3-disentanglerconv-morephylocodes.yaml -t True --gpus 0,
# python main.py --name Phylo-VQVAE --postfix 256img-phylo-vqvae-phase2-disentanglerconv-morefclayers --base configs/custom_vqgan-256emb-256img-phylo-vqvae-phase3-disentanglerconv-morefclayers.yaml -t True --gpus 0,

# python main.py --prefix /fastscratch/elhamod --name Phylo-VQVAE --postfix 256img-phase4-swapconvin-nolastrelu-multiclass --base configs/custom_vqgan-256emb-256img-phylo-vqvae-phase2-morech-8cbperlevel-morenonattr-multiclass-swapconvin-nolastrelu.yaml -t True --gpus 0,
# python main.py --prefix /fastscratch/elhamod --name Phylo-VQVAE --postfix 256img-phase4-swapconvin-nolastrelu-multiclass-256ch --base configs/custom_vqgan-256emb-256img-phylo-vqvae-phase2-morech-8cbperlevel-morenonattr-multiclass-swapconvin-nolastrelu-256ch.yaml -t True --gpus 0,
# python main.py --prefix /fastscratch/elhamod --name Phylo-VQVAE --postfix 256img-phase4-swapconvin-nolastrelu-multiclass-lowerlr --base configs/custom_vqgan-256emb-256img-phylo-vqvae-phase2-morech-8cbperlevel-morenonattr-multiclass-swapconvin-nolastrelu-lowerlr.yaml -t True --gpus 0, 

# python main.py --prefix /fastscratch/elhamod --name Phylo-VQVAE --postfix 256img-phase4-multiclass-lowerlr-4cblevel --base configs/custom_vqgan-256emb-256img-phylo-vqvae-phase4-4cbperlevel-lowerlr.yaml -t True --gpus 0, 
# python main.py --prefix /fastscratch/elhamod --name Phylo-VQVAE --postfix 256img-phase4-multiclass-lowerlr-8cblevel-stronganti --base configs/custom_vqgan-256emb-256img-phylo-vqvae-phase4-8cbperlevel-lowerlr-stronganti.yaml -t True --gpus 0, 

# python main.py --prefix /fastscratch/elhamod --name Phylo-VQVAE --postfix TESTING-256img-phase4-multiclass-lowerlr-8cblevel-stronganti --base configs/custom_vqgan-256emb-256img-phylo-vqvae-phase4-8cbperlevel-lowerlr-stronganti.yaml -t True --gpus 0,

# python main.py --prefix /fastscratch/elhamod --name Phylo-VQVAE --postfix 256img-phase4-multiclass-lowerlr-8cblevel-stronganti-morefclayers --base configs/custom_vqgan-256emb-256img-phylo-vqvae-phase4-8cbperlevel-lowerlr-stronganti_morefclayers.yaml -t True --gpus 0, 

# python main.py --prefix /fastscratch/elhamod --name Phylo-VQVAE --postfix TESTING-256img-phase4-multiclass-lowerlr-8cblevel-stronganti-largerembedding --base configs/custom_vqgan-256emb-256img-phylo-vqvae-phase4-8cbperlevel-lowerlr-stronganti-largerembedding.yaml -t True --gpus 0,

# python main.py --prefix /fastscratch/elhamod --name Phylo-VQVAE --postfix 512img-phase4-multiclass-lowerlr-8cblevel-stronganti-largerembedding --base configs/custom_vqgan-256emb-512img-phylo-vqvae-phase4-8cbperlevel-lowerlr-stronganti.yaml -t True --gpus 0,
# python main.py --prefix /fastscratch/elhamod --name Phylo-VQVAE --postfix 512img-phase4-multiclass-lowerlr-8cblevel-stronganti-2layers-cyclical --base configs/custom_vqgan-256emb-256img-phylo-vqvae-phase4-8cbperlevel-lowerlr-stronganti-1hiddenlayer.yaml -t True --gpus 0,

# python main.py --prefix /fastscratch/elhamod --name Phylo-VQVAE --postfix 512img-phase4-multiclass-lowerlr-8cblevel-stronganti-2layers-cyclical-classweighting --base configs/custom_vqgan-256emb-256img-phylo-vqvae-phase4-8cbperlevel-lowerlr-stronganti-1hiddenlayer-classweights.yaml -t True --gpus 0,

# python main.py --prefix /fastscratch/elhamod --name Phylo-VQVAE --postfix 256img-phase5-betaforclassification-phylohigher --base configs/custom_vqgan-256emb-256img-phylo-vqvae-phase5-betaforclassification-phylohigher.yaml -t True --gpus 0, 
# python main.py --prefix /fastscratch/elhamod --name Phylo-VQVAE --postfix 256img-phase5-largerbatch --base configs/custom_vqgan-256emb-256img-phylo-vqvae-phase5-largerbatch.yaml -t True --gpus 0, 

# python main.py --prefix /fastscratch/elhamod --name Phylo-VQVAE --postfix 256img-phase5-speciesout --base configs/custom_vqgan-256emb-256img-phylo-vqvae-phase5-speciesout.yaml -t True --gpus 0, 
# python main.py --prefix /fastscratch/elhamod --name Phylo-VQVAE --postfix 512img-phase5-4cbperlevel-higherlr --base configs/custom_vqgan-256emb-512img-phylo-vqvae-phase5-4cbperlevel-higherlr.yaml -t True --gpus 0, 

# python main.py --prefix /fastscratch/elhamod --name Phylo-VQVAE --postfix 512img-phase5-evenlargerbatch --base configs/custom_vqgan-256emb-256img-phylo-vqvae-phase5-evenlargerbatch.yaml -t True --gpus 0, 
# python main.py --prefix /fastscratch/elhamod --name Phylo-VQVAE --postfix 512img-phase5-morecodelocations --base configs/custom_vqgan-256emb-256img-phylo-vqvae-phase5-morecodelocations.yaml -t True --gpus 0, 

python main.py --prefix /fastscratch/elhamod --name Phylo-VQVAE --postfix 512img-phase5-lowerphyloweight --base configs/custom_vqgan-256emb-256img-phylo-vqvae-phase5-lowerphyloweight.yaml -t True --gpus 0, 

exit;




# Run these for the dataset you want before updating the custom_vqgan.yaml file and then running this script
# find /home/elhamod/data/Fish/test_padded_512 -name "*.???" > /home/elhamod/data/Fish/taming_transforms_fish_test_padded_512.txt
# find /home/elhamod/data/Fish/train_padded_512 -name "*.???" > /home/elhamod/data/Fish/taming_transforms_fish_train_padded_512.txt

# find /home/elhamod/data/Fish/test_padded_256 -name "*.???" > /home/elhamod/data/Fish/taming_transforms_fish_test_padded_256.txt
# find /home/elhamod/data/Fish/train_padded_256 -name "*.???" > /home/elhamod/data/Fish/taming_transforms_fish_train_padded_256.txt