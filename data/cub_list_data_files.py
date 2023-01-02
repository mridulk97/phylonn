import glob

train = []
for dir in glob.glob('fastscratch/mridul/cub_190_split_resized/official/CUB_200_2011/train_256/*'):
    train.append(glob.glob(str(dir)+'/*'))

flat_list_train = [item for sublist in train for item in sublist]
print('Train', len(flat_list_train))

with open('/fastscratch/mridul/cub_190_split_resized/official/CUB_200_2011/train_256.txt', 'w') as f:
    for line in flat_list_train:
        f.write(f"{line}\n")



test = []
for dir in glob.glob('/fastscratch/mridul/cub_190_split_resized/official/CUB_200_2011/test_256/*'):
    test.append(glob.glob(str(dir)+'/*'))

flat_list_test = [item for sublist in test for item in sublist]
print('Test', len(flat_list_test))

with open('/fastscratch/mridul/cub_190_split_resized/official/CUB_200_2011/test_256.txt', 'w') as f:
    for line in flat_list_test:
        f.write(f"{line}\n")