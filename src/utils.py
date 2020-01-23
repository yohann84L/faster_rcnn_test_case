import matplotlib.pyplot as plt
from random import randrange
from PIL import ImageDraw, ImageFont

class Constants:
    # Label Mapping
    LANG_LABEL = ["fr", "en"]
    COL_LABEL_NAME = "labelling_name_"
    COL_LABEL_ID = "labelling_id"


def plot_example(dataset: "FoodVisorDataset", idx: int = None):
    if not idx:
        idx = randrange(len(dataset))
    img, target = dataset[idx]

    boxes = target["boxes"]
    labels = target["labels"]
    for bbox, label in zip(boxes[labels == 1], labels[labels == 1]):
        x0, y0 = bbox[0], bbox[1]
        x1, y1 = x0 + bbox[2], y0 + bbox[3]
        draw_bbox = ImageDraw.Draw(img)
        draw_bbox.rectangle([(x0, y0), (x1, y1)], outline="red", width=2)
        draw_bbox.text((x0, y0), "Tomato", font=ImageFont.truetype("arial", 14, ), fill="red")
    img.show(title=target["image_filename"])
