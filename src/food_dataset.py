import json
import random
import re
from pathlib import Path

import albumentations as A
import numpy as np
import pandas as pd
import torch
from PIL import Image
from skimage import io
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
                # Coco dataset format : [x, y, width, height]
                boxes.append(obj["box"])
                label_str = self.__get_label_for_id(obj["id"])
                label = self.__is_aliment_present(label_str)
                labels.append(label)
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.float32)
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        image_id = torch.tensor([index])

        target = {
            "boxes": boxes,
            "labels": labels,
            "area": area,
            "image_id": image_id,
            "image_filename": img_id
        }

        if self.transforms:
            img = self.transforms(image=io.imread(img_name))["image"]
        else:
            img = Image.fromarray(io.imread(img_name))

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

def split_train_test_valid_json(
    img_annotation_path: str,
    random_seed: int = None,
    split_size=(
        0.8,
        0.2)):
    """
    Method to split dataset ids into train, test and valid

    Argument:
    ---------
        - img_annotation_path (str): filename/path of the annotation json file
        - random_seed (int): seed of the random shuffle
        - split_size (tuple of float): float size for each split

    Return:
    -------
        - 2 or 3 dictionary corresponding to the splitted annotation json file
    """
    with open(img_annotation_path) as f:
        img_annotation = json.load(f)

    img_ids = list(img_annotation.keys())
    if random_seed:
        random.seed(random_seed)
    random.shuffle(img_ids)
    total_length = len(img_ids)
    img_ids = np.array(img_ids)

    if len(split_size) == 1 and split_size[0] <= 1:
        split_key = np.split(img_ids, [np.floor(total_length * split_size[0])])
        return (
            {k: v for k, v in img_annotation.items() if k in split_key[0]},
            {k: v for k, v in img_annotation.items() if k in split_key[1]},
        )
    elif len(split_size) == 2 and split_size[0] <= 1:
        split_key = np.split(img_ids,
                             [int(np.floor(total_length * split_size[0]))])
        return (
            {k: v for k, v in img_annotation.items() if k in split_key[0]},
            {k: v for k, v in img_annotation.items() if k in split_key[1]},
        )
    elif len(split_size) == 3 and split_size[0] <= 1:
        split_key = np.split(
            img_ids,
            [
                int(np.floor(total_length * split_size[0])),
                int(np.floor(total_length * (split_size[0] + split_size[1]))),
            ],
        )
        return (
            {k: v for k, v in img_annotation.items() if k in split_key[0]},
            {k: v for k, v in img_annotation.items() if k in split_key[1]},
            {k: v for k, v in img_annotation.items() if k in split_key[2]},
        )

    else:
        raise NotImplementedError
