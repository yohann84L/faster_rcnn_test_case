from src.food_dataset import FoodVisorDataset
from src.utils.metric_logger import MetricLogger, SmoothedValue
import src.utils as utils
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
import math
import sys
import time
from src.model import FasterRCNNFood


def load_agumentation_pipelines():
    # Define the augmentation pipeline
    augmentation_pipeline_train = A.Compose(
        [
            A.Resize(width=512, height=512),
            A.HorizontalFlip(p=0.5),  # apply horizontal flip to 50% of images
            A.Rotate(
                limit=90, p=0.5
            ),  # apply random with limit of 90° to 50% of images
            A.OneOf(
                [
                    # apply one of transforms to 30% of images
                    A.RandomBrightnessContrast(),  # apply random contrast & brightness
                    A.RandomGamma(),  # apply random gamma
                ],
                p=0.3,
            ),
            A.OneOf(
                [
                    # apply one of transforms to 30% images
                    A.ElasticTransform(
                        alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03
                    ),
                    A.GridDistortion(),
                    A.OpticalDistortion(distort_limit=2, shift_limit=0.5),
                ],
                p=0.3,
            ),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),  # convert the image to PyTorch tensor
        ],
        p=1,
    )

    # Define the transformation pipeline for test
    tranformation_pipeline_test = A.Compose(
        [
            A.Resize(width=512, height=512),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),  # convert the image to PyTorch tensor
        ],
        p=1,
    )

    return augmentation_pipeline_train, tranformation_pipeline_test

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

    train_transforms, test_transforms = load_agumentation_pipelines()
    dataset = build_dataset(train_transforms)
    dataset_test = build_dataset(test_transforms)

    # split the dataset in train and test set
    indices = torch.randperm(len(dataset)).tolist()
    size = int(len(indices) * 0.15)
    dataset = torch.utils.data.Subset(dataset, indices[:-size])
    dataset_test = torch.utils.data.Subset(dataset_test, indices[-size:])

    # define training and validation data loaders
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=2, shuffle=True, num_workers=4,
        collate_fn=utils.collate_fn)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, shuffle=False, num_workers=4,
        collate_fn=utils.collate_fn)

    print("Start training")
    model.train(data_loader, data_loader_test, num_epochs=3, use_cuda=True)



