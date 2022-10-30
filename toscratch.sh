


mkdir -p /fastscratch/elhamod/data/Fish
cp -avr /home/elhamod/data/Fish/taming_transforms_fish_test_padded_256.txt /fastscratch/elhamod/data/Fish/taming_transforms_fish_test_padded_256.txt
cp -avr /home/elhamod/data/Fish/taming_transforms_fish_train_padded_256.txt /fastscratch/elhamod/data/Fish/taming_transforms_fish_train_padded_256.txt
cp -avr /home/elhamod/data/Fish/taming_transforms_fish_test_padded_512.txt /fastscratch/elhamod/data/Fish/taming_transforms_fish_test_padded_512.txt
cp -avr /home/elhamod/data/Fish/taming_transforms_fish_train_padded_512.txt /fastscratch/elhamod/data/Fish/taming_transforms_fish_train_padded_512.txt
cp -avr /home/elhamod/data/Fish/cleaned_metadata.tre /fastscratch/elhamod/data/Fish/cleaned_metadata.tre
cp -avr /home/elhamod/data/Fish/name_conversion.pkl /fastscratch/elhamod/data/Fish/name_conversion.pkl
cp -avr /home/elhamod/data/Fish/test_padded_256 /fastscratch/elhamod/data/Fish/test_padded_256
cp -avr /home/elhamod/data/Fish/train_padded_256 /fastscratch/elhamod/data/Fish/train_padded_256
cp -avr /home/elhamod/data/Fish/test_padded_512 /fastscratch/elhamod/data/Fish/test_padded_512
cp -avr /home/elhamod/data/Fish/train_padded_512 /fastscratch/elhamod/data/Fish/train_padded_512

mkdir -p /fastscratch/elhamod/projects/taming-transformers/logs/512pixels_512embedding
mkdir -p /fastscratch/elhamod/projects/taming-transformers/logs/512pixels_256embedding
mkdir -p /fastscratch/elhamod/projects/taming-transformers/logs/256pixels_256embedding
mkdir -p  /fastscratch/elhamod/projects/taming-transformers/SLURM
cp -avr /home/elhamod/projects/taming-transformers/logs/256pixels_256embedding/checkpoints /fastscratch/elhamod/projects/taming-transformers/logs/256pixels_256embedding/checkpoints
cp -avr /home/elhamod/projects/taming-transformers/logs/512pixels_256embedding/checkpoints /fastscratch/elhamod/projects/taming-transformers/logs/512pixels_256embedding/checkpoints
cp -avr /home/elhamod/projects/taming-transformers/logs/512pixels_512embedding/checkpoints /fastscratch/elhamod/projects/taming-transformers/logs/512pixels_512embedding/checkpoints
cp -avr /home/elhamod/projects/taming-transformers/logs/256pixels_256embedding/configs /fastscratch/elhamod/projects/taming-transformers/logs/256pixels_256embedding/configs
cp -avr /home/elhamod/projects/taming-transformers/logs/512pixels_256embedding/configs /fastscratch/elhamod/projects/taming-transformers/logs/512pixels_256embedding/configs
cp -avr /home/elhamod/projects/taming-transformers/logs/512pixels_512embedding/configs /fastscratch/elhamod/projects/taming-transformers/logs/512pixels_512embedding/configs