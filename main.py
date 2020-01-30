import torch

import src.utils as utils
from src.datasets import FoodVisorDataset, DatasetTransforms
from src.model import FasterRCNNFood
from src.utils.visualization import plot_prediction, plot_example


def build_dataset(dataset_params: dict, transforms: DatasetTransforms) -> FoodVisorDataset:
    return FoodVisorDataset(
        json_annotations="configs/img_annotations.json",
        csv_mapping="configs/label_mapping.csv",
        imgs_folder="data/",
        regex_aliment="[Tt]omate(s)?",
        transforms=transforms
    )


if __name__ == '__main__':
    # plot_example(dataset, idx=1270)
    model = FasterRCNNFood(
        backbone_name="resnet50_32_4d",
        pretrained=True,
        num_classes=2
    )

    dataset = build_dataset({}, DatasetTransforms(train=True))
    dataset_test = build_dataset({}, DatasetTransforms(train=False))

    # plot_example(dataset, 23)

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

    #model.load_model_for_inference("models/test_model.pth", cuda=False)
    #img, pred = model.predict(dataset, 23)
    #plot_prediction(img, pred)
