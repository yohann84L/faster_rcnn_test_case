import colorsys
import random

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import torch

from .utils import unormalize_tensor


def plot_example(dataset: "FoodVisorDataset", idx: int = None):
    if not idx:
        idx = random.randrange(len(dataset))
    img, target = dataset[idx]

    img = unormalize_tensor(img)

    fig, ax = plt.subplots(1)
    ax.imshow(img.numpy().transpose(1, 2, 0))
    boxes = target["boxes"]
    labels = target["labels"]
    for bbox, label in zip(boxes[labels == 1], labels[labels == 1]):
        x0, y0, x1, y1 = bbox
        rect = patches.Rectangle((x0, y0), x1 - x0, y1 - y0, linewidth=2,
                              alpha=0.7, linestyle="dashed",
                              edgecolor=(1, 0, 0), facecolor='none')
        # Add the patch to the Axes
        ax.add_patch(rect)

        # Label
        label = "Tomato"
        ax.text(x0, y0 + 8, label,
                color='w', size=11, backgroundcolor="none")
    plt.show()


def plot_prediction(img: torch.Tensor, pred: dict):
    img = unormalize_tensor(img)
    fig, ax = plt.subplots(1)
    ax.imshow(img.numpy().transpose(1, 2, 0))
    boxes = pred["boxes"]
    labels = pred["labels"]
    for bbox, label in zip(boxes[labels == 1], labels[labels == 1]):
        x0, y0, x1, y1 = bbox
        rect = patches.Rectangle((x0, y0), x1 - x0, y1 - y0, linewidth=2,
                              alpha=0.7, linestyle="dashed",
                              edgecolor=(1, 0, 0), facecolor='none')
        # Add the patch to the Axes
        ax.add_patch(rect)

        # Label
        score = 1
        label = "Tomato"
        caption = "{} {:.3f}".format(label, score) if score else label
        ax.text(x0, y0 + 8, caption,
                color='w', size=11, backgroundcolor="none")
    plt.show()


def random_colors(N, bright=True):
    """
    Generate random colors.
    To get visually distinct colors, generate them in HSV space then
    convert to RGB.
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.shuffle(colors)
    return colors
