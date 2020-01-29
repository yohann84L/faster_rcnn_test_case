import json
import re
from pathlib import Path
from typing import Union

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import SubsetRandomSampler

from src.utils import Constants


class FoodVisorDataset(torch.utils.data.Dataset):
    """
    FoodVisorDataset is a custom Dataset. It is adapted to FoodVisor data convention.
    Indeed, to build this dataset, à img_annotations and a csv_mapping are needed.

    Arguments:
    ----------
        - json_annotations (dict): dictionnary of the img_annotations.json
        - csv_mapping (str): path file for the label_mapping.csv
        - img_folder (str): path folder where all images are located
        - regex_aliment (str): regex to build class. Example: with regex r"[Tt]omate(s)?" with build two classes,
        one containing only image with tomatoes, and one with everything else.
        - augmentations (albumentation, default=None): Transform to apply using albumentation
        - lang (str, default="fr"): lang corresponding to label ("fr" and "en" only)
    """

    def __init__(
            self,
            json_annotations: str,
            csv_mapping: str,
            imgs_folder: str,
            regex_aliment: str,
            transforms: A = None,
            lang: str = "fr"):
        self.imgs_folder = Path(imgs_folder)
        with open(Path(json_annotations).as_posix()) as f:
            self.img_annotations = json.load(f)
        self.csv_mapping = pd.read_csv(csv_mapping)

        self.transforms = transforms
        self.__regex_aliment = regex_aliment
        if lang in Constants.LANG_LABEL:
            self.__lang = lang
        else:
            print("lang parameter should be one of the following :")
            for l in Constants.LANG_LABEL:
                print("   - {:s}".format(l))
            raise ValueError

    def __getitem__(self, index: int):
        img_id = list(self.img_annotations.keys())[index]
        img_name = self.imgs_folder / img_id

        objs = self.img_annotations[img_id]
        boxes = []
        labels = []
        for obj in objs:
            if not obj["is_background"]:
                boxes.append(coco_to_pascalvoc(obj["box"]))
                label_str = self.__get_label_for_id(obj["id"])
                label = self.__is_aliment_present(label_str)
                labels.append(label)

        img = cv2.imread(img_name.as_posix())
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.transforms:
            data = {
                "image": img,
                "bboxes": boxes,
                "labels": labels}
            res = self.transforms(**data)
            img = res["image"]
            boxes = res["bboxes"]

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        area = boxes[:, 3] * boxes[:, 2]
        image_id = torch.tensor([index])

        target = {
            "boxes": boxes,
            "labels": labels,
            "area": area,
            "image_id": image_id,
            # "image_filename": img_id
        }

        return img, target

    def __len__(self) -> int:
        return len(self.img_annotations.keys())

    def __get_label_for_id(self, label_id: str) -> str:
        """
        Method to get the label from a label id using the label mapping

        Argument:
        ---------
            - label_id (str): id of the label
        Return:
        -------
            - label (str)
        """
        return self.csv_mapping[self.csv_mapping[Constants.COL_LABEL_ID]
                                == label_id][Constants.COL_LABEL_NAME + self.__lang].values[0]

    def __is_aliment_present(self, label: str) -> int:
        """
        Method to check if an aliment is present in an image.

        Argument:
        ---------
            - image_id (str): id of the image we want to check
        Return:
        -------
            - boolean: true if the image contains aliment, false else
        """
        if bool(re.search(self.__regex_aliment, label)):
            return 1
        else:
            return 0


def coco_to_pascalvoc(bbox: Union[list, tuple, np.array]) -> Union[list, tuple, np.array]:
    # Coco dataset format : [x, y, width, height]
    x, y, w, h = bbox[0], bbox[1], bbox[2] - 1, bbox[3] - 1
    # Transform into pascal_voc format : [xmin, ymin, xmax, ymax]
    x0, y0 = min(x, x + w), min(y, y + h)
    x1, y1 = max(x, x + w), max(y, y + h)
    return x0, y0, x1, y1
