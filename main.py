import albumentations as A
import cv2
import torch
from albumentations.pytorch import ToTensorV2 as ToTensor

import src.utils as utils
from src.food_dataset import FoodVisorDataset
from src.model import FasterRCNNFood
from src.utils import plot_example

class DatasetTransforms:
    def __init__(self, train=True, bbox_param=None):
        self.train = train
        if not bbox_param:
            bbox_param = A.BboxParams(
                format="pascal_voc",
                label_fields=["labels"])
        self.bbox_param = bbox_param
        self.transforms = []

        # Add base tranform
        self.add_transforms()
        # Add normalization
        self.add_normalization()
        # Convert with ToTensor()
        self.transforms.append(ToTensor())

    def __call__(self, **data):
        return A.Compose(self.transforms, self.bbox_param)(**data)

    def add_transforms(self):
        if self.train:
            self.transforms += [
                A.Resize(256, 256),
                A.RandomCrop(224, 244),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.Rotate(p=0.5, border_mode=cv2.BORDER_REFLECT, value=0)
            ]
        else:
            self.transforms += [
                A.Resize(224, 244),
            ]

    def add_normalization(self):
        self.transforms += [
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
        ]



def build_dataset(transforms):
    dataset = FoodVisorDataset(
        json_annotations="configs/img_annotations.json",
        csv_mapping="configs/label_mapping.csv",
        imgs_folder="data/",
        regex_aliment="[Tt]omate(s)?",
        transforms=transforms
    )

    return dataset

if __name__ == '__main__':
    # plot_example(dataset, idx=1270)
    model = FasterRCNNFood(
        pretrained=True,
        num_classes=2
    )

    dataset = build_dataset(DatasetTransforms(train=True))
    dataset_test = build_dataset(DatasetTransforms(train=False))

    #plot_example(dataset, 23)

    # split the dataset in train and test set
    indices = torch.randperm(len(dataset)).tolist()
    size = int(len(indices) * 0.15)
    dataset = torch.utils.data.Subset(dataset, indices[:-size])
    dataset_test = torch.utils.data.Subset(dataset_test, indices[-size:])

    # define training and validation data loaders
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=8, shuffle=True, num_workers=4,
        collate_fn=utils.collate_fn)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, shuffle=False, num_workers=4,
        collate_fn=utils.collate_fn)

    print("Start training")
    model.train(data_loader, data_loader_test, num_epochs=3, use_cuda=True)



