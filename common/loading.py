from os import PathLike

import torch
from torchvision import datasets
from torchvision.transforms import ToTensor, Normalize, Compose
import logging

import models
from models import baseClassifier

from typing import TypedDict, NewType, Literal
from enum import Enum, auto


class Datasets(Enum):
    CIFAR100 = auto()


class Models(Enum):
    SimpleCNN = auto()


ValidDataset = NewType("ValidDatasets", Datasets)
ModelTypes = NewType("ModelTypes", Models)


class DsKwargs(TypedDict, total=False):
    data_set: ValidDataset | str
    root: PathLike


class ModelKwargs(TypedDict, total=False):
    type: ModelTypes | str
    num_classes: int
    load: Literal[False] | PathLike


def load_dataset(ds_kwargs: DsKwargs) -> torch.utils.data.Dataset:
    ds_type = ds_kwargs.pop("data_set")
    data_set: datasets = getattr(datasets, ds_type)
    transforms = Compose((ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))))
    data_set = data_set(train=True, transform=transforms, **ds_kwargs)
    return data_set


def load_model(
    model_kwargs: ModelKwargs, ds_input_size: tuple[int, ...]
) -> baseClassifier:
    using_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {using_device}")
    m_type = model_kwargs.pop("type")
    model = getattr(models, m_type)
    model: models.BaseClassifier = model(
        **model_kwargs, input_shape=ds_input_size, device=using_device
    )
    return model
