
alias python=python3.8
# CUDA_VISIBLE_DEVICES=5,6,7 
python main.py --name Phylo-VQVAE --base configs/custom_vqgan-256emb-512img-phylo-vqvae-lambdapgml.yaml -t True --gpus 5,6,7

exit;


# Run these for the dataset you want before updating the custom_vqgan.yaml file and then running this script
#find /raid/elhamod/Fish/phylo-VQVAE/test_padded_512/ -name "*.???" > /raid/elhamod/Fish/phylo-VQVAE/taming_transforms_fish_test_padded_512.txt
#find /raid/elhamod/Fish/phylo-VQVAE/train_padded_512/ -name "*.???" > /raid/elhamod/Fish/phylo-VQVAE/taming_transforms_fish_train_padded_512.txt
