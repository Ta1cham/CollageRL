import imgsim
import torch

import sys

sys.path.append('/home/taogawa/modify_collage')
from module.dataloader import DataLoader
import cv2
import random
import os
import warnings
warnings.filterwarnings("ignore")

def choice_image_from_imagenet(path):
    target_dirs = os.listdir(os.path.expanduser(path))
    targe_dir = random.choice(target_dirs)
    target_images = os.listdir(os.path.expanduser(f'{path}/{targe_dir}'))
    target_image = random.choice(target_images)
    return cv2.imread(os.path.expanduser(f'{path}/{targe_dir}/{target_image}'))

def choice_image_from_dataloader(dataloader):
    image, label = dataloader.get_random_source('train')
    return image

def test_imgsim(img1, img2):
    vtr = imgsim.Vectorizer()
    vec0 = vtr.vectorize(img1)
    vec1 = vtr.vectorize(img2)
    dist = imgsim.distance(vec0, vec1) 
    return dist


def main():
    path = '~/Datasets/imagenet/train'
    dataloader = DataLoader(data_path='~/Datasets', goal='imagenet', source='imagenet', width=224, height=224, debug=False)

    img1 = choice_image_from_imagenet(path)
    img1 = torch.tensor(img1)
    img2 = choice_image_from_dataloader(dataloader)
    img2 = img2.astype('uint8')

    print(img1.shape)
    print(img1.dtype)
    print(img2.shape)
    print(img2.dtype)

    dist = test_imgsim(img1, img2)

if __name__ == '__main__':
    main()