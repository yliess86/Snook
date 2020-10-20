import numpy as np
import torch
import torch.nn as nn

from PIL import Image
from snook.config import AttributeDict
from snook.data.dataset.dir import DirDataset
from snook.model.network.dir import DirNet
from snook.model.train.base import Trainer
from torch.utils.data import DataLoader
from torch.optim import Adam
from tqdm import tqdm


class DirSyntheticTrainer(Trainer):
    def __init__(self, model: AttributeDict, trainer: AttributeDict) -> None:
        
        print("[DirNet][Trainer][Synthetic] Preparing Training Data")
        self.train_dataset = DirDataset(trainer.dataset.train_render, trainer.dataset.train_data, train=True)
        self.train_loader = DataLoader(self.train_dataset, batch_size=trainer.hyperparameters.batch_size, shuffle=True, pin_memory=True, num_workers=4)

        print("[DirNet][Trainer][Synthetic] Preparing Validation Data")
        self.valid_dataset = DirDataset(trainer.dataset.valid_render, trainer.dataset.valid_data, train=False)
        self.valid_loader = DataLoader(self.valid_dataset, batch_size=trainer.hyperparameters.batch_size, shuffle=False, pin_memory=True, num_workers=4)

        print("[DirNet][Trainer][Synthetic] Preparing Testing  Data")
        self.test_dataset = DirDataset(trainer.dataset.test_render, trainer.dataset.test_data, train=False)
        self.test_loader = DataLoader(self.test_dataset, batch_size=trainer.hyperparameters.batch_size, shuffle=False, pin_memory=True, num_workers=4)

        print(f"[DirNet][Trainer][Synthetic] Preparing Model and Optimizer")
        self.model = DirNet.from_config(model).cuda()
        self.optim = Adam(self.model.parameters(), lr=trainer.hyperparameters.lr, weight_decay=trainer.hyperparameters.weight_decay)
        self.loss = nn.MSELoss().cuda()

    def train(self) -> None:
        self.model = self.model.train()

        total_loss = 0.0
        pbar = tqdm(self.train_loader, desc="[DirNet][Trainer][Synthetic] Training")
        for img, dir in pbar:
            img, dir = img.cuda(), dir.cuda()

            self.optim.zero_grad()

            _dir = self.model(img)
            loss = self.loss(_dir, dir)

            loss.backward()
            self.optim.step()
            
            total_loss += loss.item()
            pbar.set_postfix(loss=total_loss / len(self.train_loader))

    def valid(self) -> None:
        self.model = self.model.eval()

        total_loss = 0.0
        pbar = tqdm(self.valid_loader, desc="[DirNet][Trainer][Synthetic] Validation")
        with torch.no_grad():
            for img, dir in pbar:
                img, dir = img.cuda(), dir.cuda()

                _dir = self.model(img)
                loss = self.loss(_dir, dir)
                
                total_loss += loss.item()
                pbar.set_postfix(loss=total_loss / len(self.valid_loader))

    def test(self) -> None:
        self.model = self.model.eval()

        total_loss = 0.0
        pbar = tqdm(self.test_loader, desc="[DirNet][Trainer][Synthetic] Testing")
        with torch.no_grad():
            for img, dir in pbar:
                img, dir = img.cuda(), dir.cuda()

                _dir = self.model(img)
                loss = self.loss(_dir, dir)
                
                total_loss += loss.item()
                pbar.set_postfix(loss=total_loss / len(self.test_loader))

    def __call__(self, epochs: int, debug: bool = False) -> None:
        print("[DirNet][Trainer][Synthetic] Training")
        for epoch in range(epochs):
            print(f"[DirNet][Trainer][Synthetic] Training Epoch [{epoch + 1:0{len(str(epochs))}d}/{epochs}]")
            self.train()
            self.valid()
        self.test()