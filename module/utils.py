import torch

def create_mono(color, weight, height):
    return torch.ones(1, 3, height, height) * color

def create_canvas(args, device, goal):
    mode = args.canvas_setup
    if mode == "white":
        return torch.ones_like(goal)
    elif mode == "random":
        return torch.rand_like(goal)
    elif mode == "complement_all":
        canvas = 