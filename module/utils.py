import numpy as np
import cv2
import argparse
import datetime
import os

def create_mono(color, weight, height):
    color_r, color_g, color_b = color
    canvas = np.zeros((weight, height, 3))
    canvas[:, :, 0] = color_r
    canvas[:, :, 1] = color_g
    canvas[:, :, 2] = color_b
    return canvas

def create_canvas(args, goal):
    mode = args.canvas_setup
    if mode == "white":
        return np.ones_like(goal)
    elif mode == "random":
        return np.random.rand(*goal.shape)
    elif mode == "complement":
        color_goal = np.mean(goal, axis=(0, 1))
        color_canvas = 1 - color_goal
        canvas = create_mono(color_canvas, *goal.shape[:2])
        return canvas
    elif mode == "msemax":
        color_goal = np.mean(goal, axis=(0, 1))
        color_canvas = 1 - np.round(color_goal)
        print(color_canvas)
        canvas = create_mono(color_canvas, *goal.shape[:2])
        return canvas
    elif mode == "rev_comp":
        canvas = 1 - goal
        return canvas
    elif mode == "rev_msemax":
        canvas = np.round(1 - goal)
        return canvas
    # for test
    elif mode == "mean":
        color_goal = np.mean(goal, axis=(0, 1))
        print(color_goal)
        canvas = create_mono(color_goal, *goal.shape[:2])
        return canvas

def test_create_canvas():
    parser = argparse.ArgumentParser()
    parser.add_argument("--canvas_setup", type=str, default="white")
    args = parser.parse_args()

    path = "../samples/goals/boat.jpg"
    goal = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)/255
    goal_origin = np.uint8(goal*255)

    # white
    canvas_white = create_canvas(args, goal)
    canvas_white = np.uint8(canvas_white*255)

    # mean
    args.canvas_setup = "mean"
    canvas_mean = create_canvas(args, goal)
    canvas_mean = np.uint8(canvas_mean*255)

    # random
    args.canvas_setup = "random"
    canvas_random = create_canvas(args, goal)
    canvas_random = np.uint8(canvas_random*255)

    # complement
    args.canvas_setup = "complement"
    canvas_complement = create_canvas(args, goal)
    canvas_complement = np.uint8(canvas_complement*255)

    # msemax
    args.canvas_setup = "msemax"
    canvas_msemax = create_canvas(args, goal)
    canvas_msemax = np.uint8(canvas_msemax*255)

    # rev_comp
    args.canvas_setup = "rev_comp"
    canvas_rev_comp = create_canvas(args, goal)
    canvas_rev_comp = np.uint8(canvas_rev_comp*255)

    # rev_msemax
    args.canvas_setup = "rev_msemax"
    canvas_rev_msemax = create_canvas(args, goal)
    canvas_rev_msemax = np.uint8(canvas_rev_msemax*255)

    result = np.concatenate([goal_origin, canvas_mean, canvas_white, canvas_random, canvas_complement, canvas_msemax, canvas_rev_comp, canvas_rev_msemax], axis=0)
    time = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    if not os.path.exists("./test"):
        print(os.getcwd())
        raise FileNotFoundError("Directory not found")
    cv2.imwrite(f"./test/result_{time}.png", cv2.cvtColor(result, cv2.COLOR_RGB2BGR))


if __name__ == "__main__":
    test_create_canvas()