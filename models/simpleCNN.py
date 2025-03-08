from pathlib import Path

import torch
from torch import nn
from torch.nn import Sequential
from torchvision.datasets import CIFAR100
from torchvision.transforms import ToTensor, Normalize, Compose
import logging
from functools import reduce

from models.baseClassifier import BaseClassifier

NUM_CLASSES = 100
NUM_WORKERS = 2
DATA_PATH = r"D:\cifar100"
BATCH_SIZE = 64

logging.basicConfig(level=logging.DEBUG)
logger = logging


class SimpleCNN(BaseClassifier):
    head: Sequential

    def __specific_init__(
        self, input_shape: tuple[int, ...], num_classes: int, device: torch.device
    ):
        self.backbone = nn.ModuleList()
        self.backbone.append(nn.Conv2d(self.input_shape[0], 4, kernel_size=4, stride=2))
        for i in range(6):
            in_ch = 2 ** (i + 2)
            self.backbone.append(
                nn.Conv2d(in_ch, 2 * in_ch, kernel_size=3, padding="valid")
            )
            self.backbone.append(nn.ReLU())

        init_tensor = torch.zeros((1, *input_shape))
        logger.debug(f"Initial test tensor shape: {init_tensor.shape}")
        init_tensor = self.backbone_forward(init_tensor)
        logger.debug(f"Test tensor after backbone shape: {init_tensor.shape}")
        logger.debug(
            f"Flat feature size: {reduce(lambda x, y: x * y, init_tensor.shape[1:])}"
        )

        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(reduce(lambda x, y: x * y, init_tensor.shape[1:]), 256),
            nn.ReLU(),
            nn.Linear(256, self.num_classes),
            nn.LogSoftmax(dim=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone_forward(x)
        y = self.head(x)
        return y

    def backbone_forward(self, x):
        for l in self.backbone:
            x = l(x)
        return x


if __name__ == "__main__":
    using_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {using_device}")
    data_path = Path(DATA_PATH)
    transforms = Compose((ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))))
    ds = CIFAR100(train=True, transform=transforms, root=data_path)
    dl = torch.utils.data.DataLoader(
        ds,
        BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
    )
    model = SimpleCNN(ds[0][0].size(), NUM_CLASSES, using_device)
    model.to(model.device)
    model.fit(5, dl)
