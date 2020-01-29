import albumentations as A
from albumentations.pytorch import ToTensorV2 as ToTensor
from cv2 import BORDER_REFLECT


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
                A.Rotate(p=0.5, border_mode=BORDER_REFLECT, value=0)
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
