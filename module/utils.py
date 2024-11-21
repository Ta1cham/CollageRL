import torch
import numpy as np
import cv2
import argparse
import datetime
import os

def create_mono(color, weight, height):
    color_r, color_g, color_b = color
    canvas = torch.zeros(weight, height, 3)
    canvas[:, :, 0] = color_r
    canvas[:, :, 1] = color_g
    canvas[:, :, 2] = color_b
    return canvas

def create_canvas(args, device, goal):
    mode = args.canvas_setup
    if mode == "white":
        return torch.ones_like(goal)
    elif mode == "random":
        return torch.rand_like(goal)
    elif mode == "complement":
        color_goal = torch.mean(goal, dim=(0, 1))
        color_canvas = 1 - color_goal
        canvas = create_mono(color_canvas, *goal.shape[:2])
        return canvas
    elif mode == "msemax":
        color_goal = torch.mean(goal, dim=(0, 1))
        color_canvas = 1 - torch.round(color_goal, decimals=0)
        print(color_canvas)
        canvas = create_mono(color_canvas, *goal.shape[:2])
        return canvas
    # for test
    elif mode == "mean":
        color_goal = torch.mean(goal, dim=(0, 1))
        print(color_goal)
        canvas = create_mono(color_goal, *goal.shape[:2])
        return canvas

def test_create_canvas():
    parser = argparse.ArgumentParser()
    parser.add_argument("--canvas_setup", type=str, default="white")
    args = parser.parse_args()

    device = "cpu"
    path = "../../Datasets/imagenet/train/n01494475/n01494475_5.JPEG"
    goal = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)/255
    goal_origin = np.uint8(goal*255)
    goal = torch.tensor(goal).to(device)

    # white
    canvas_white = create_canvas(args, device, goal)
    canvas_white = np.uint8(canvas_white.numpy()*255)

    # mean
    args.canvas_setup = "mean"
    canvas_mean = create_canvas(args, device, goal)
    canvas_mean = np.uint8(canvas_mean.numpy()*255)

    # random
    args.canvas_setup = "random"
    canvas_random = create_canvas(args, device, goal)
    canvas_random = np.uint8(canvas_random.numpy()*255)

    # complement
    args.canvas_setup = "complement"
    canvas_complement = create_canvas(args, device, goal)
    canvas_complement = np.uint8(canvas_complement.numpy()*255)

    # msemax
    args.canvas_setup = "msemax"
    canvas_msemax = create_canvas(args, device, goal)
    canvas_msemax = np.uint8(canvas_msemax.numpy()*255)

    result = np.concatenate([goal_origin, canvas_mean, canvas_white, canvas_random, canvas_complement, canvas_msemax], axis=0)
    time = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    if not os.path.exists("./test"):
        print(os.getcwd())
        raise FileNotFoundError("Directory not found")
    cv2.imwrite(f"./test/result_{time}.png", cv2.cvtColor(result, cv2.COLOR_RGB2BGR))


if __name__ == "__main__":
    test_create_canvas()