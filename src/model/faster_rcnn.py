import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator

from .engine import train_one_epoch, evaluate


class FasterRCNNFood:
    def __init__(self, backbone_name, pretrained=True, finetune=True, num_classes=2):
        # # load model
        # self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=pretrained)
        #
        # # define number of classe and set the output of the classifier to this number
        # in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        # self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

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

    def train(self, data_loader, data_loader_test, num_epochs=10, use_cuda=True, plot_running=True):

        # choose device
        if use_cuda and torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

        # define dataset
        self.model.to(device)
        params = [p for p in self.model.parameters() if p.requires_grad]
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

        writer = SummaryWriter()

        for epoch in range(num_epochs):
            # train for one epoch, printing every 50 iterations
            train_one_epoch(self.model, optimizer, data_loader, device, epoch, print_freq=50, writer=writer)
            # update the learning rate
            lr_scheduler.step()
            # evaluate on the test dataset
            evaluate(self.model, data_loader_test, device=device, writer=writer, epoch=epoch)
        writer.close()
        print("That's it!")

    def save_model(self, path):
        torch.save(self.model.state_dict(), "test_model.pth")

    def load_model_for_inference(self, path, pretrained=True, num_classes=2, cuda=True):
        self.__init__(pretrained, num_classes)
        device = torch.device("cuda") if (cuda and torch.cuda.is_available()) else torch.device("cpu")
        self.model.load_state_dict(torch.load(path, map_location=device))
        self.model.eval()

    def predict(self, dataset, idx):
        img, _ = dataset[idx]
        img.to("cpu")
        self.model.eval()
        self.model.to("cpu")
        pred = self.model([img])
        print(pred)
        return img, pred[0]


def build_backbone(base_model_name, pretrained, finetune):
    base_model_accepted = [
        "mobilenetv2",
        "vgg16",
        "resnext50_32_4d"
    ]

    # Mobilenet v2
    if base_model_name == "mobilenetv2":
        backbone = torchvision.models.mobilenet_v2(pretrained).features
        backbone.out_channels = 1280
    # VGG 16
    elif base_model_name == "vgg16":
        backbone = torchvision.models.vgg16(pretrained).features
        set_grad_for_finetunning(backbone, 10)
        backbone.out_channels = 512
    # Resnext 50
    elif base_model_name == "resnext50":
        backbone = torch.nn.Sequential(*list(torchvision.models.resnext50_32x4d(pretrained).children())[:-2])
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
