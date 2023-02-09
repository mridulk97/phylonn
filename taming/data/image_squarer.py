import os
import PIL
import glob
import argparse
import numpy as np
from tqdm import tqdm
from torchvision import transforms


def MakeSquared(img, imageDimension=256, padding='white'):

    img_H = img.size[0]
    img_W = img.size[1]
    smaller_dimension = 0 if img_H < img_W else 1
    larger_dimension = 1 if img_H < img_W else 0
    if (imageDimension != img_H or imageDimension != img_W):
        new_smaller_dimension = int(imageDimension * img.size[smaller_dimension] / img.size[larger_dimension])
        if smaller_dimension == 1:
            img = transforms.functional.resize(img, (new_smaller_dimension, imageDimension))
        else:
            img = transforms.functional.resize(img, (imageDimension, new_smaller_dimension))

        diff = imageDimension - new_smaller_dimension
        pad_1 = int(diff/2)
        pad_2 = diff - pad_1

        if padding == 'imagenet':
            mean = np.asarray([ 0.485, 0.456, 0.406 ])
            fill = tuple([int(round(mean[0]*255)), int(round(mean[1]*255)), int(round(mean[2]*255))])
        elif padding=='black':
            fill = tuple([0, 0, 0])
        else:
            fill = tuple([255, 255, 255])

        if smaller_dimension == 0:
            img = transforms.functional.pad(img, (pad_1, 0, pad_2, 0), padding_mode='constant', fill = fill)
        else:
            img = transforms.functional.pad(img, (0, pad_1, 0, pad_2), padding_mode='constant', fill = fill)

    return img


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='List cub data files')
    parser.add_argument('--image_dir', type=str, help='Path of images to be resized')
    parser.add_argument('--output_dir', type=str, help='Path of resized images')
    parser.add_argument('--padding', type=str, help='Padding type - either white, black or imagenet mean')

    args = parser.parse_args()

    for folder in tqdm(sorted(glob.glob(os.path.join(args.image_dir, '*')))):
        species = folder.split('/')[-1]
        new_folder = os.path.join(args.output_dir, species)
        if not os.path.isdir(new_folder):
            os.makedirs(new_folder)
        
        for image_path in sorted(glob.glob(folder+'/*')):
            image_file = image_path.split('/')[-1]
            image = PIL.Image.open(image_path)
            try:
                squared_image = MakeSquared(image, padding=args.padding)
                new_image_path = os.path.join(new_folder, image_file)
                squared_image.save(new_image_path)
            except Exception as error:
                print(image_path)
                print(error)