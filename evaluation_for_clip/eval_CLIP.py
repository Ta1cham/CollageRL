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
import nvidia_smi



def get_device():
    nvidia_smi.nvmlInit()
    device_Count = nvidia_smi.nvmlDeviceGetCount()
    freem = []
    for i in range(device_Count):
        handle = nvidia_smi.nvmlDeviceGetHandleByIndex(i)
        info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
        freem.append(info.free)
    nvidia_smi.nvmlShutdown()
    device_num = np.argmax(freem).item()
    torch.cuda.set_device(device_num)
    device = torch.device('cuda:{}'.format(device_num))
    return device

def get_targets(path):
    path = os.path.expanduser(path)
    dirs = os.listdir(path)

    return dirs
    

def load_images(path):
    """
    preprocessing = Compose([
        Resize(224), #interpolation=InterpolationMode.BICUBIC),
        CenterCrop(224),
        ToTensor()
    ])
    """
    # {image_name}/{type_bg}.jpg
    dirs = get_targets(path)
    contents_dict = {}
    for dir in tqdm(dirs):
        images = {}
        files = os.listdir(os.path.join(path, dir))
        for file in files:
            type_bg = file.split(".")[0]
            img = Image.open(os.path.join(path, dir, file))
            images[type_bg] = img
        contents_dict[dir] = images
    return contents_dict

def calc_similarities(content_dict, device):
    model, preprocess = clip.load("ViT-B/32", device=device)
    text = ["a human made collage image"]
    scores = {'white': [], 'random': [], 'rev_comp': [], 'rev_msemax': []}
    for name, images in content_dict.items():
        for type_bg, img in images.items():
            image_input = preprocess(img).unsqueeze(0).to(device)
            text_input = clip.tokenize(text).to(device)
            with torch.no_grad():
                image_features = model.encode_image(image_input)
                text_features = model.encode_text(text_input)

                logits_per_image, logits_per_text = model(image_input, text_input)
            scores[type_bg].append(logits_per_image.item() / 100)
            if (logits_per_image.item() / 100) < 0.2:
                print("  ", name, type_bg, logits_per_image.item() / 100)
    print(len(scores['white']))
    print(len(scores['random']))
    print(len(scores['rev_comp']))
    print(len(scores['rev_msemax']))
    return scores

def main():
    path = "~/modify_collage/evaluation_for_clip/results"
    path = os.path.expanduser(path)
    device = get_device()
    contents_dict = load_images(path)
    scores = calc_similarities(contents_dict, device)
    print("white: mean: ", np.array(scores['white']).mean(), "std: ", np.array(scores['white']).std(), "min: ", np.array(scores['white']).min(), "max: ", np.array(scores['white']).max())
    print("random: mean: ", np.array(scores['random']).mean(), "std: ", np.array(scores['random']).std(), "min: ", np.array(scores['random']).min(), "max: ", np.array(scores['random']).max())
    print("rev_comp: mean: ", np.array(scores['rev_comp']).mean(), "std: ", np.array(scores['rev_comp']).std(), "min: ", np.array(scores['rev_comp']).min(), "max: ", np.array(scores['rev_comp']).max())
    print("rev_msemax: mean: ", np.array(scores['rev_msemax']).mean(), "std: ", np.array(scores['rev_msemax']).std(), "min: ", np.array(scores['rev_msemax']).min(), "max: ", np.array(scores['rev_msemax']).max())


if __name__ == '__main__':
    main()