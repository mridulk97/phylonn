#!/bin/bash

# python main.py --name Phylo-VQVAE --base configs/custom_vqgan-256emb-512img-phylo-vqvae-afterhyperp.yaml -t True --gpus 0, #1 # crashes... not enough memory?!
# python main.py --name Phylo-VQVAE --base configs/custom_vqgan-256emb-512img-phylo-vqvae.yaml -t True --gpus 0, #1
# python main.py --name Phylo-VQVAE --base configs/custom_vqgan-256emb-512img-phylo-vqvae-phyloloss.yaml -t True --gpus 0,
# python test_ground_2.py --name LSF2-VQVAE --postfix translation --base configs/custom_vqgan-1024emb-256img-LSF2-vqvae-inference.yaml --classification_config analysis/configs/test_resnet_all_levels.yaml -t True --gpus 0,
python translateLSF.py --name LSF2-VQVAE --postfix translation --base configs/custom_vqgan-1024emb-256img_imagenetmean_noaug-batch5-LSF2-vqvae-base_noaug_withimgnet-inference.yaml -t True --gpus 0,

exit;




# Run these for the dataset you want before updating the custom_vqgan.yaml file and then running this script
# find /home/elhamod/data/Fish/test_padded_512 -name "*.???" > /home/elhamod/data/Fish/taming_transforms_fish_test_padded_512.txt
# find /home/elhamod/data/Fish/train_padded_512 -name "*.???" > /home/elhamod/data/Fish/taming_transforms_fish_train_padded_512.txt

# find /home/elhamod/data/Fish/test_padded_256 -name "*.???" > /home/elhamod/data/Fish/taming_transforms_fish_test_padded_256.txt
# find /home/elhamod/data/Fish/train_padded_256 -name "*.???" > /home/elhamod/data/Fish/taming_transforms_fish_train_padded_256.txt