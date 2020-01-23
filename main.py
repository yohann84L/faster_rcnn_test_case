from src.utils import plot_example
from src.food_dataset import FoodVisorDataset
import src.utils as utils
import torchvision
import torch
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
import albumentations as A
from albumentations.pytorch import ToTensorV2
import math
import sys


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

def build_model():
    # load pretrained model
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

    # define number of classe and set the output of the classifier to this number
    num_classes = 2
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # backbone model
    backbone = torchvision.models.mobilenet_v2(pretrained=True).features
    backbone.out_channels = 1280

    anchor_generator = AnchorGenerator(
        sizes=((32, 64, 128, 256, 512), ),
        aspect_ratios=((0.5, 1.0, 2.0), )
    )

    roi_pooler = torchvision.ops.MultiScaleRoIAlign(
        featmap_names=['0'],
        output_size=7,
        sampling_ratio=2
    )

    model = FasterRCNN(
        backbone=backbone,
        num_classes=num_classes,
        rpn_anchor_generator=anchor_generator,
        box_roi_pool=roi_pooler
    )

    return model


def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    lr_scheduler = None
    if epoch == 0:
        warmup_factor = 1. / 1000
        warmup_iters = min(1000, len(data_loader) - 1)

        lr_scheduler = utils.warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor)

    for images, targets in metric_logger.log_every(data_loader, print_freq, header):
        images = list(image.to(device) for image in images)
        print(targets)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)

        losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        loss_value = losses_reduced.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])


def train(model, data_loader):
    # choose device
    device = torch.device("cpu") if torch.cuda.is_available() else torch.device("cpu")

    # define dataset
    model.to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params=params,
        lr=0.005,
        momentum=0.9,
        weight_decay=0.0005
    )

    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer=optimizer,
        step_size=3,
        gamma=0.1
    )

    num_epochs = 10

    # let's train it for 10 epochs
    num_epochs = 10

    for epoch in range(num_epochs):
        # train for one epoch, printing every 10 iterations
        train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
        # update the learning rate
        lr_scheduler.step()
        # evaluate on the test dataset
        #evaluate(model, data_loader_test, device=device)

    print("That's it!")

if __name__ == '__main__':
    # plot_example(dataset, idx=1270)
    model = build_model()

    train_transforms, test_transforms = load_agumentation_pipelines()
    dataset = build_dataset(train_transforms)
    dataset_test = build_dataset(test_transforms)

    # split the dataset in train and test set
    indices = torch.randperm(len(dataset)).tolist()
    size = int(len(indices) * 0.15)
    print(size)
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
    train(model, data_loader)



