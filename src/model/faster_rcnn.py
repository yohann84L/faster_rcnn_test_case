from datetime import datetime
from pathlib import Path
from typing import Union

import torch
import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator

from .engine import train_one_epoch, evaluate


class FasterRCNNFood:
    def __init__(self, backbone_name: str, pretrained: bool = True, finetune: bool = True, num_classes: int = 2):
        self.__pretrained = pretrained
        self.__num_classes = num_classes
        self.__model_name = backbone_name
        backbone = build_backbone(backbone_name, pretrained, finetune)

        anchor_generator = AnchorGenerator(
            sizes=((32, 64, 128, 256, 512),),
            aspect_ratios=((0.5, 1.0, 2.0),)
        )

        roi_pooler = torchvision.ops.MultiScaleRoIAlign(
            featmap_names=[0],
            output_size=7,
            sampling_ratio=2
        )

        self.model = FasterRCNN(
            backbone=backbone,
            num_classes=num_classes,
            rpn_anchor_generator=anchor_generator,
            box_roi_pool=roi_pooler
        )

        self.params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.Adam(
            params=self.params,
            lr=0.005,
            weight_decay=0.0005
        )

        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer=self.optimizer,
            step_size=3,
            gamma=0.1
        )

    def train(self, data_loader: DataLoader, data_loader_test: DataLoader, num_epochs: int = 10, use_cuda: bool = True,
              epoch_save_ckpt: Union[int, list] = None, dir: str = None):
        """
        Method to train FasterRCNNFood model.
        Args:
            data_loader (torch.utils.data.DataLoader): data loader to train model on
            data_loader_test (torch.utils.data.DataLoader): data loader to evaluate model on
            num_epochs (int = 10): number of epoch to train model
            use_cuda (bool = True): use cuda or not
            epoch_save_ckpt (list or int): Epoch at which you want to save the model. If -1 save only last epoch.
            dir (str = "models/): Directory where model are saved under the name "{model_name}_{date}_ep{epoch}.pth"
        """
        if epoch_save_ckpt == -1:
            epoch_save_ckpt = [num_epochs - 1]
        if not dir:
            dir = "models"
        dir = Path(dir)
        dir.mkdir(parents=True, exist_ok=True)
        # choose device
        if use_cuda and torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

        # define dataset
        self.model.to(device)
        writer = SummaryWriter()

        for epoch in range(num_epochs):
            # train for one epoch, printing every 50 iterations
            train_one_epoch(self.model, self.optimizer, data_loader, device, epoch, print_freq=50, writer=writer)
            # update the learning rate
            self.lr_scheduler.step()
            # evaluate on the test dataset
            evaluate(self.model, data_loader_test, device=device, writer=writer, epoch=epoch)
            # save checkpoint
            if epoch in epoch_save_ckpt:
                self.save_checkpoint(dir.as_posix(), epoch)
        writer.close()
        print("That's it!")

    def save_checkpoint(self, dir: str, epoch: int):
        """
        Save a model checkpoint at a given epoch.
        Args:
            dir: dir folder to save the .pth file
            epoch: epoch the model is
        """
        state = {'epoch': epoch + 1,
                 'state_dict': self.model.state_dict(),
                 'optimizer': self.optimizer.state_dict(),
                 'num_classes': self.__num_classes,
                 'pretrained': self.__pretrained,
                 "model_name": self.__model_name}
        now = datetime.now()
        filename = "{model_name}_{date}_ep{epoch}.pth".format(
            model_name=self.__model_name,
            date=now.strftime("%b%d_%H-%M"),
            epoch=epoch
        )
        torch.save(state, Path(dir) / filename)
        "Checkpoint saved : {}".format(Path(dir) / filename)

    def predict(self, dataset, idx):
        img, _ = dataset[idx]
        img.to("cpu")
        self.model.eval()
        self.model.to("cpu")
        pred = self.model([img])
        return img, pred[0]

    @staticmethod
    def load_checkpoint(filename: str, cuda: bool = True) -> ("FasterRCNNFood", int):
        """
        Load a model checkpoint to continue training.
        Args:
            filename (str): filename/path of the checkpoint.pth
            cuda (bool = True): use cuda

        Returns:
            (FasterRCNNFood) model
            (int) number of epoch + 1 the model was trained with
        """
        device = torch.device("cuda") if (cuda and torch.cuda.is_available()) else torch.device("cpu")
        start_epoch = 0
        if Path(filename).exists():
            print("=> loading checkpoint '{}'".format(filename))
            checkpoint = torch.load(filename, map_location=device)
            # Load params
            pretrained = checkpoint['pretrained']
            num_classes = checkpoint["num_classes"]
            start_epoch = checkpoint['epoch']
            model_name = checkpoint['model_name']
            # Build model key/architecture
            model = FasterRCNNFood(model_name, pretrained, num_classes)
            # Update model and optimizer
            model.model.load_state_dict(checkpoint['state_dict'])
            model.optimizer.load_state_dict(checkpoint['optimizer'])

            model.model = model.model.to(device)
            # now individually transfer the optimizer parts...
            for state in model.optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(device)

            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(filename, checkpoint['epoch']))
            return model, start_epoch
        else:
            print("=> no checkpoint found at '{}'".format(filename))

    @staticmethod
    def load_for_inference(filename: str, cuda: bool = True) -> "FasterRCNNFood":
        """
        Load a model checkpoint to make inference.
        Args:
            filename (str): filename/path of the checkpoint.pth
            cuda (bool = True): use cuda
        Returns:
            (FasterRCNNFood) model
        """
        device = torch.device("cuda") if (cuda and torch.cuda.is_available()) else torch.device("cpu")
        if Path(filename).exists():
            print("=> loading checkpoint '{}'".format(filename))
            checkpoint = torch.load(filename, map_location=device)
            # Load params
            pretrained = checkpoint['pretrained']
            num_classes = checkpoint["num_classes"]
            model_name = checkpoint['model_name']
            # Build model key/architecture
            model = FasterRCNNFood(model_name, pretrained, num_classes)
            # Update model and optimizer
            model.model.load_state_dict(checkpoint['state_dict'])
            model.model = model.model.to(device)
            model.model = model.model.eval()

            print("=> loaded checkpoint '{}'".format(filename))
            return model
        else:
            print("=> no checkpoint found at '{}'".format(filename))


def build_backbone(base_model_name, pretrained, finetune):
    base_model_accepted = [
        "mobilenetv2",
        "vgg16",
        "resnet18",
        "resnet50",
        "resnext50"
    ]

    # Mobilenet v2
    if base_model_name == "mobilenetv2":
        backbone = torchvision.models.mobilenet_v2(pretrained).features
        backbone.out_channels = 1280
    # VGG 16
    elif base_model_name == "vgg16":
        backbone = torchvision.models.vgg16(pretrained).features
        if finetune:
            set_grad_for_finetunning(backbone, 10)
        backbone.out_channels = 512
    # ResNet 18
    elif base_model_name == "resnet18":
        backbone = torch.nn.Sequential(*list(torchvision.models.resnet18(pretrained).children())[:-2])
        if finetune:
            set_grad_for_finetunning(backbone, 7)
        backbone.out_channels = 512
    # ResNet 50
    elif base_model_name == "resnext50":
        backbone = torch.nn.Sequential(*list(torchvision.models.resnet50(pretrained).children())[:-2])
        if finetune:
            set_grad_for_finetunning(backbone, 7)
        backbone.out_channels = 2048
    # ResNext 50
    elif base_model_name == "resnext50":
        backbone = torch.nn.Sequential(*list(torchvision.models.resnext50_32x4d(pretrained).children())[:-2])
        if finetune:
            set_grad_for_finetunning(backbone, 7)
        backbone.out_channels = 2048
    else:
        print("Backbone model should be one of the following list: ")
        for name in base_model_accepted:
            print("     - {}".format(name))
        raise NotImplementedError
    return backbone


def set_grad_for_finetunning(backbone, layer_number):
    count = 0
    for child in backbone.children():
        count += 1
        if count < layer_number:
            for param in child.parameters():
                param.requires_grad = False
