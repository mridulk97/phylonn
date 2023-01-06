import os
import glob
import argparse

def create_text_file(image_dir, output_file):
    file_list = []
    for dir in glob.glob(os.path.join(image_dir, '*')):
        file_list.append(glob.glob(str(dir)+'/*'))

    file_list_flattened = [item for sublist in file_list for item in sublist]
    print('Number of images:', len(file_list_flattened))

    with open(output_file, 'w') as f:
        for line in file_list_flattened:
            f.write(f"{line}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='List cub data files')
    parser.add_argument('--image_dir', type=str, help='Path of images to create list files')
    parser.add_argument('--output_file', type=str, help='Path of text file to be created')

    args = parser.parse_args()
    create_text_file(args.image_dir, args.output_file)