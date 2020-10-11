import numpy as np
import torch
import torch.nn.functional as F

from PIL import Image
from snook.config import AttributeDict
from snook.data.dataset.label import LabelDataset
from snook.model.network.label import LabelNet
from snook.model.train.base import Trainer
from torch.utils.data import DataLoader
from torch.optim import Adam
from tqdm import tqdm
from typing import Tuple


class LabelSyntheticTrainer(Trainer):
    def __init__(self, model: AttributeDict, trainer: AttributeDict) -> None:
        print("[LabelNet][Trainer][Synthetic] Preparing Training Data")
        self.train_dataset = LabelDataset(trainer.dataset.train_render, trainer.dataset.train_data, train=True)
        self.train_loader = DataLoader(self.train_dataset, batch_size=trainer.hyperparameters.batch_size, shuffle=True, pin_memory=True, num_workers=4)

        print("[LabelNet][Trainer][Synthetic] Preparing Validation Data")
        self.valid_dataset = LabelDataset(trainer.dataset.valid_render, trainer.dataset.valid_data, train=False)
        self.valid_loader = DataLoader(self.valid_dataset, batch_size=trainer.hyperparameters.batch_size, shuffle=False, pin_memory=True, num_workers=4)

        print("[LabelNet][Trainer][Synthetic] Preparing Testing  Data")
        self.test_dataset = LabelDataset(trainer.dataset.test_render, trainer.dataset.test_data, train=False)
        self.test_loader = DataLoader(self.test_dataset, batch_size=trainer.hyperparameters.batch_size, shuffle=False, pin_memory=True, num_workers=4)

        print("[LabelNet][Trainer][Synthetic] Preparing Model and Optimizer")
        self.model = LabelNet.from_config(model).cuda()
        self.optim = Adam(self.model.parameters(), lr=trainer.hyperparameters.lr, weight_decay=trainer.hyperparameters.weight_decay)

    def train(self) -> None:
        self.model = self.model.train()

        total_loss = 0.0
        total_acc = 0.0
        pbar = tqdm(self.train_loader, desc="[LabelNet][Trainer][Synthetic] Training")
        for render, label in pbar:
            render, label = render.cuda(), label.cuda()

            self.optim.zero_grad()

            _log_probs = self.model(render)
            loss = F.nll_loss(_log_probs, label)
            acc = (torch.argmax(_log_probs, dim=1) == label).float().sum()

            loss.backward()
            self.optim.step()
            
            total_loss += loss.item()
            total_acc += acc.item()
            pbar.set_postfix(loss=total_loss / len(self.train_loader), acc=f"{total_acc / len(self.train_dataset) * 100:.2f}%")

    def valid(self) -> None:
        self.model = self.model.eval()

        total_loss = 0.0
        total_acc = 0.0
        pbar = tqdm(self.valid_loader, desc="[LabelNet][Trainer][Synthetic] Validation")
        with torch.no_grad():
            for render, label in pbar:
                render, label = render.cuda(), label.cuda()

                _log_probs = self.model(render)
                loss = F.nll_loss(_log_probs, label)
                acc = (torch.argmax(_log_probs, dim=1) == label).float().sum()
                
                total_loss += loss.item()
                total_acc += acc.item()
                pbar.set_postfix(loss=total_loss / len(self.valid_loader), acc=f"{total_acc / len(self.valid_dataset) * 100:.2f}%")

    def test(self) -> None:
        self.model = self.model.eval()

        total_loss = 0.0
        total_acc = 0.0
        pbar = tqdm(self.test_loader, desc="[LabelNet][Trainer][Synthetic] Testing")
        with torch.no_grad():
            for render, label in pbar:
                render, label = render.cuda(), label.cuda()

                _log_probs = self.model(render)
                loss = F.nll_loss(_log_probs, label)
                acc = (torch.argmax(_log_probs, dim=1) == label).float().sum()
                
                total_loss += loss.item()
                total_acc += acc.item()
                pbar.set_postfix(loss=total_loss / len(self.test_loader), acc=f"{total_acc / len(self.test_dataset) * 100:.2f}%")

    def __call__(self, epochs: int) -> None:
        print("[LabelNet][Trainer][Synthetic] Training")
        for epoch in range(epochs):
            print(f"[LabelNet][Trainer][Synthetic] Training Epoch [{epoch + 1:0{len(str(epochs))}d}/{epochs}]")
            self.train()
            self.valid()
        self.test()