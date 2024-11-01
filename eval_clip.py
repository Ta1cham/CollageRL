import clip
import torchvision
import cv2
from PIL import Image
import os
import random
import numpy as np
import torch
from tqdm import tqdm
from torchvision.transforms import Compose, Resize, CenterCrop, Normalize, ToTensor
from torchvision.transforms import InterpolationMode
import matplotlib.pyplot as plt

def get_device():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    return device

def get_targets(path):
    path = os.path.expanduser(path)
    dirs = os.listdir(path)
    

def load_images_random(path, device):
    preprocessing = Compose([
        Resize(224), #interpolation=InterpolationMode.BICUBIC),
        CenterCrop(224),
        ToTensor()
    ])
    images = []
    for dir in tqdm(dirs):
        files = os.listdir(os.path.join(path, dir))
        files_sample = random.sample(files, 10)
        for file in files_sample:
            #image = cv2.imread(os.path.join(path, dir, file))
            #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = Image.open(os.path.join(path, dir, file)).convert('RGB')
            image = preprocessing(image)
            images.append(image)
    
    images = torch.tensor(np.stack(images))
    image_mean = images.mean(dim=(0, 2, 3))
    image_std = images.std(dim=(0, 2, 3))
    images = (images - image_mean[None, :, None, None]) / image_std[None, :, None, None]

    return images.to(device)

def calculate_similarity(images, device):
    model, preprocess = clip.load("ViT-B/32", device=device)

    #set text
    text = "man made collage"
    text_input = clip.tokenize(text).to(device)
    print("text_input.shape: ", text_input.shape)

    with torch.no_grad():
        image_features = model.encode_image(images).float()
        text_features = model.encode_text(text_input).float()
        text_features /= text_features.norm(dim=-1, keepdim=True)
    text_probs = torch.cosine_similarity(image_features, text_features)

    print('image_features.shape = ', image_features.shape)
    print('text_features.shape = ', text_features.shape)
    print('text_probs.shape = ', text_probs.shape)

    probs = np.argsort(text_probs.cpu(), axis=0)
    figs, ax = plt.subplots(5, 10, figsize=(50, 100))
    for i in range(5):
        for j in range(5):
            idx = probs[i*5+j]
            img = np.array(images[idx].cpu().permute(1, 2, 0).shape)
            ax[i, j].imshow(img)
    
    plt.

            



def main():
    path = '~/Datasets/imagenet/train'
    device = get_device()
    images = load_images_random(path, device)
    calculate_similarity(images, device)


if __name__ == '__main__':
    main()
