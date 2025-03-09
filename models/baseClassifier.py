import logging
from abc import ABCMeta, abstractmethod
from pathlib import Path
from typing import Literal, Optional

import torch
import tqdm
from torch import nn
from torch.nn import Module
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


logging.basicConfig(level=logging.DEBUG)
logger = logging


class BaseClassifier(Module, metaclass=ABCMeta):
    def __init__(
        self,
        input_shape: tuple[int, ...],
        num_classes: int,
        device: torch.device,
        load: Path | Literal[False] = False,
        optimizer: dict = {},
    ):
        super().__init__()
        self.device = device
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.epoch = 0
        self.loss = nn.NLLLoss()
        self.__specific_init__(input_shape, num_classes, device)
        self.optimizer = torch.optim.AdamW(self.parameters(), **optimizer)

        self.to(self.device)
        if load is not False:
            load: Path
            self.load(load)
            logger.info(f"Loading weights from: {load}")

    def __specific_init__(
        self, input_shape: tuple[int, ...], num_classes: int, device: torch.device
    ):
        pass

    @abstractmethod
    def forward(self, x):
        pass

    def train_batch(self, x, y) -> tuple[torch.Tensor, torch.Tensor]:
        self.optimizer.zero_grad()
        y_pred = self.forward(x)
        loss = self.loss(y_pred, y)
        loss.backward()
        self.optimizer.step()
        correct = (y == y_pred.argmax(dim=1)).sum().item()
        return loss.item(), correct / y.shape[0]

    def run_epoch(
        self, data_loader: DataLoader, train: bool = True
    ) -> tuple[torch.Tensor, torch.Tensor]:
        loss_history = []
        acc_history = []
        pbar = tqdm.tqdm(data_loader, position=0, leave=False)
        pbar_str = "Batch: " if train else "Validation: "
        pbar.set_description_str(pbar_str)
        for x, y in pbar:
            x: torch.Tensor
            y: torch.Tensor
            x = x.to(self.device)
            y = y.to(self.device)
            if train:
                b_loss, b_acc = self.train_batch(x, y)
            else:
                b_loss, b_acc = self.eval_batch(x, y)
            loss_history.append(b_loss)
            acc_history.append(b_acc)
            pbar.set_postfix_str(f"batch loss {b_loss:.2f}")
        loss_history = torch.tensor(loss_history)
        acc_history = torch.tensor(acc_history)
        return loss_history, acc_history

    def fit(
        self,
        num_epoch: int,
        data_loader: DataLoader,
        writer: Optional[SummaryWriter] = None,
        test_data_loader: Optional[DataLoader] = None,
    ):
        pbar = tqdm.trange(num_epoch, position=1)
        pbar.set_description_str("Epochs: ")
        for _ in pbar:
            e_loss, e_acc = self.run_epoch(data_loader)
            self.epoch += 1
            mean_loss = e_loss.mean().item()
            mean_acc = e_acc.mean().item()
            pbar.set_postfix_str(f"epoch loss: {mean_loss :.2f}")
            if writer is not None:
                writer.add_scalar("Loss/Train", mean_loss, self.epoch)
                writer.add_scalar("Accuracy/Train", mean_acc, self.epoch)

            if test_data_loader is not None:
                v_loss, v_acc = self.run_epoch(test_data_loader, train=False)
                mean_v_loss = v_loss.mean().item()
                mean_v_acc = v_acc.mean().item()
                if writer is not None:
                    writer.add_scalar("Loss/Validation", mean_v_loss, self.epoch)
                    writer.add_scalar("Accuracy/Validation", mean_v_acc, self.epoch)

    def save(self, destination: str | Path):
        torch.save(
            {
                "epoch": self.epoch,
                "model_state_dict": self.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                # "loss": loss,
            },
            destination,
        )

    def load(self, path: Path):
        checkpoint = torch.load(path, weights_only=True, map_location=self.device)
        self.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.epoch = checkpoint["epoch"]
        self.to(self.device)

    def eval_batch(
        self, x: torch.Tensor, y: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self.eval()
        with torch.no_grad():
            y_pred = self.forward(x)
            loss = self.loss(y_pred, y)
            correct = (y == y_pred.argmax(dim=1)).sum().item()

            self.train()
            return loss.item(), correct / y.shape[0]
