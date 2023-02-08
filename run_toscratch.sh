


mkdir -p /fastscratch/elhamod/data/Fish
cp -avr /home/elhamod/data/Fish/fish_test_padded_256.txt /fastscratch/elhamod/data/Fish/fish_test_padded_256.txt
cp -avr /home/elhamod/data/Fish/fish_train_padded_256.txt /fastscratch/elhamod/data/Fish/fish_train_padded_256.txt
cp -avr /home/elhamod/data/Fish/fish_test_padded_512.txt /fastscratch/elhamod/data/Fish/fish_test_padded_512.txt
cp -avr /home/elhamod/data/Fish/fish_train_padded_512.txt /fastscratch/elhamod/data/Fish/fish_train_padded_512.txt
cp -avr /home/elhamod/data/Fish/cleaned_metadata.tre /fastscratch/elhamod/data/Fish/cleaned_metadata.tre
cp -avr /home/elhamod/data/Fish/name_conversion.pkl /fastscratch/elhamod/data/Fish/name_conversion.pkl
cp -avr /home/elhamod/data/Fish/test_padded_256 /fastscratch/elhamod/data/Fish/test_padded_256
cp -avr /home/elhamod/data/Fish/train_padded_256 /fastscratch/elhamod/data/Fish/train_padded_256
cp -avr /home/elhamod/data/Fish/test_padded_512 /fastscratch/elhamod/data/Fish/test_padded_512
cp -avr /home/elhamod/data/Fish/train_padded_512 /fastscratch/elhamod/data/Fish/train_padded_512

cp -avr /home/elhamod/data/Fish/fish_test_padded_256_out.txt /fastscratch/elhamod/data/Fish/fish_test_padded_256_out.txt
cp -avr /home/elhamod/data/Fish/fish_train_padded_256_out.txt /fastscratch/elhamod/data/Fish/fish_train_padded_256_out.txt

cp -avr /home/elhamod/projects/phylonn/logs/2022-11-07T15-02-16_Phylo-VQVAE256img-phase5-nolastrelumlp /fastscratch/elhamod/logs/2022-11-07T15-02-16_Phylo-VQVAE256img-phase5-nolastrelumlp

cp -avr /home/elhamod/projects/phylonn/logs/2022-11-04T15-27-23_Phylo-VQVAE256img-phase4-nophylo /fastscratch/elhamod/logs/2022-11-04T15-27-23_Phylo-VQVAE256img-phase4-nophylo


mkdir -p /fastscratch/elhamod/projects/phylonn/logs/512pixels_512embedding
mkdir -p /fastscratch/elhamod/projects/phylonn/logs/512pixels_256embedding
mkdir -p /fastscratch/elhamod/projects/phylonn/logs/256pixels_256embedding
mkdir -p  /fastscratch/elhamod/projects/phylonn/SLURM
cp -avr /home/elhamod/projects/phylonn/logs/256pixels_256embedding/checkpoints /fastscratch/elhamod/projects/phylonn/logs/256pixels_256embedding/checkpoints
cp -avr /home/elhamod/projects/phylonn/logs/512pixels_256embedding/checkpoints /fastscratch/elhamod/projects/phylonn/logs/512pixels_256embedding/checkpoints
cp -avr /home/elhamod/projects/phylonn/logs/512pixels_512embedding/checkpoints /fastscratch/elhamod/projects/phylonn/logs/512pixels_512embedding/checkpoints
cp -avr /home/elhamod/projects/phylonn/logs/256pixels_256embedding/configs /fastscratch/elhamod/projects/phylonn/logs/256pixels_256embedding/configs
cp -avr /home/elhamod/projects/phylonn/logs/512pixels_256embedding/configs /fastscratch/elhamod/projects/phylonn/logs/512pixels_256embedding/configs
cp -avr /home/elhamod/projects/phylonn/logs/512pixels_512embedding/configs /fastscratch/elhamod/projects/phylonn/logs/512pixels_512embedding/configs