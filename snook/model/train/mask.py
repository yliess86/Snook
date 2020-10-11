import numpy as np
import torch

from PIL import Image
from snook.config import AttributeDict
from snook.data.dataset.mask import MaskDataset
from snook.model.loss import L2Loss
from snook.model.network.mask import MaskNet
from snook.model.train.base import Trainer
from torch.utils.data import DataLoader
from torch.optim import Adam
from tqdm import tqdm


class MaskSyntheticTrainer(Trainer):
    def __init__(self, model: AttributeDict, trainer: AttributeDict) -> None:      
        print("[MaskNet][Trainer][Synthetic] Preparing Training Data")
        self.train_dataset = MaskDataset(trainer.dataset.train_render, trainer.dataset.train_data, train=True)
        self.train_loader = DataLoader(self.train_dataset, batch_size=trainer.hyperparameters.batch_size, shuffle=True, pin_memory=True, num_workers=4)

        print("[MaskNet][Trainer][Synthetic] Preparing Validation Data")
        self.valid_dataset = MaskDataset(trainer.dataset.valid_render, trainer.dataset.valid_data, train=False)
        self.valid_loader = DataLoader(self.valid_dataset, batch_size=trainer.hyperparameters.batch_size, shuffle=False, pin_memory=True, num_workers=4)

        print("[MaskNet][Trainer][Synthetic] Preparing Testing  Data")
        self.test_dataset = MaskDataset(trainer.dataset.test_render, trainer.dataset.test_data, train=False)
        self.test_loader = DataLoader(self.test_dataset, batch_size=trainer.hyperparameters.batch_size, shuffle=False, pin_memory=True, num_workers=4)

        print(f"[MaskNet][Trainer][Synthetic] Preparing Model and Optimizer")
        self.model = MaskNet.from_config(model).cuda()
        self.optim = Adam(self.model.parameters(), lr=trainer.hyperparameters.lr, weight_decay=trainer.hyperparameters.weight_decay)
        self.loss = L2Loss().cuda()

    def train(self) -> None:
        self.model = self.model.train()

        total_loss = 0.0
        pbar = tqdm(self.train_loader, desc="[MaskNet][Trainer][Synthetic] Training")
        for render, mask in pbar:
            render, mask = render.cuda(), mask.cuda()

            self.optim.zero_grad()

            _mask = self.model(render)
            loss = self.loss(_mask, mask)

            loss.backward()
            self.optim.step()
            
            total_loss += loss.item()
            pbar.set_postfix(loss=total_loss / len(self.train_loader))

    def valid(self) -> None:
        self.model = self.model.eval()

        total_loss = 0.0
        pbar = tqdm(self.valid_loader, desc="[MaskNet][Trainer][Synthetic] Validation")
        with torch.no_grad():
            for render, mask in pbar:
                render, mask = render.cuda(), mask.cuda()

                _mask = self.model(render)
                loss = self.loss(_mask, mask)
                
                total_loss += loss.item()
                pbar.set_postfix(loss=total_loss / len(self.valid_loader))

    def test(self) -> None:
        self.model = self.model.eval()

        total_loss = 0.0
        pbar = tqdm(self.test_loader, desc="[MaskNet][Trainer][Synthetic] Testing")
        with torch.no_grad():
            for render, mask in pbar:
                render, mask = render.cuda(), mask.cuda()

                _mask = self.model(render)
                loss = self.loss(_mask, mask)
                
                total_loss += loss.item()
                pbar.set_postfix(loss=total_loss / len(self.test_loader))

    def debug(self, epoch_pbar: str) -> None:
        self.model = self.model.eval()

        with torch.no_grad():
            render, mask = self.test_dataset[0]
            _mask  = self.model(render.unsqueeze(0).cuda()).cpu()
            _mask  =             _mask.permute((1, 2, 0)).numpy()
            render =            render.permute((1, 2, 0)).numpy()
            mask   = mask.unsqueeze(0).permute((1, 2, 0)).numpy()
        
        img = np.hstack([render, mask.repeat(3, -1), _mask.repeat(3, -1)]).clip(0, 1)
        img = Image.fromarray((img * 255).astype(np.uint8))
        img.show(title=f"Epoch {epoch_pbar}")

    def __call__(self, epochs: int, debug: bool = False) -> None:
        print("[MaskNet][Trainer][Synthetic] Training")
        for epoch in range(epochs):
            epoch_pbar = f"[{epoch + 1:0{len(str(epochs))}d}/{epochs}]"
            print(f"[MaskNet][Trainer][Synthetic] Training Epoch {epoch_pbar}")
            self.train()
            self.valid()
            if debug:
                self.debug(epoch_pbar)
        self.test()