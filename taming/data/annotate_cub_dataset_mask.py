import os
import PIL
from PIL import Image
import glob
import argparse
from tqdm import tqdm



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='List cub data files')
    parser.add_argument('--image_dir', type=str, help='Path of images to be resized')
    parser.add_argument('--output_dir', type=str, help='Path of resized images')
    parser.add_argument('--segmentation_base_dir', type=str, help='Path of segmentation images')

    args = parser.parse_args()
    output_dir = '/fastscratch/mridul/cub_190_split_resized/official/CUB_200_2011/test_seg'


    for folder in tqdm(sorted(glob.glob(os.path.join(args.image_dir, '*')))):
        species = folder.split('/')[-1]
        new_folder = os.path.join(args.output_dir, species)
        if not os.path.isdir(new_folder):
            os.makedirs(new_folder)
        for image_path in sorted(glob.glob(folder+'/*')):
            image_file = image_path.split('/')[-1]
            segmentation_path = os.path.join(args.segmentation_base_dir, species, image_file)
            segmentation_path = segmentation_path.replace('jpg', 'png')
            try:
                
                image = PIL.Image.open(image_path)
                segmentation_mask = PIL.Image.open(segmentation_path)
                blank = image.point(lambda _: 0)
                masked_image = PIL.Image.composite(image, blank, segmentation_mask)

                # white background
                white_background = Image.new("RGB", image.size, (255, 255, 255))
                white_background.paste(image, (0,0), segmentation_mask)
                masked_image = white_background

                new_image_path = os.path.join(new_folder, image_file)
                masked_image.save(new_image_path)

            except Exception as error:
                print(image_path)
                print(error)