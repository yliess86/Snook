import numpy as np
import torch

from PIL import Image
from snook.config import AttributeDict
from snook.data.dataset.loc import LocDataset
from snook.model.loss import AdaptiveWingLoss
from snook.model.loss import L2Loss
from snook.model.network.loc import LocNet
from snook.model.train.base import Trainer
from torch.utils.data import DataLoader
from torch.optim import Adam
from tqdm import tqdm


class LocSyntheticTrainer(Trainer):
    def __init__(self, model: AttributeDict, trainer: AttributeDict) -> None:
        self.steps = trainer.hyperparameters.steps
        self.spread = trainer.hyperparameters.spread
        self.mask = trainer.hyperparameters.mask
        
        print("[LocNet][Trainer][Synthetic] Preparing Training Data")
        self.train_dataset = LocDataset(trainer.dataset.train_render, trainer.dataset.train_data, spread=self.spread.end, train=True)
        self.train_loader = DataLoader(self.train_dataset, batch_size=trainer.hyperparameters.batch_size, shuffle=True, pin_memory=True, num_workers=4)

        print("[LocNet][Trainer][Synthetic] Preparing Validation Data")
        self.valid_dataset = LocDataset(trainer.dataset.valid_render, trainer.dataset.valid_data, spread=self.spread.end, train=False)
        self.valid_loader = DataLoader(self.valid_dataset, batch_size=trainer.hyperparameters.batch_size, shuffle=False, pin_memory=True, num_workers=4)

        print("[LocNet][Trainer][Synthetic] Preparing Testing  Data")
        self.test_dataset = LocDataset(trainer.dataset.test_render, trainer.dataset.test_data, spread=self.spread.end, train=False)
        self.test_loader = DataLoader(self.test_dataset, batch_size=trainer.hyperparameters.batch_size, shuffle=False, pin_memory=True, num_workers=4)

        print(f"[LocNet][Trainer][Synthetic] Preparing Model and Optimizer loss={trainer.hyperparameters.loss}")
        self.model = LocNet.from_config(model).cuda()
        self.optim = Adam(self.model.parameters(), lr=trainer.hyperparameters.lr, weight_decay=trainer.hyperparameters.weight_decay)
        self.loss = AdaptiveWingLoss() if trainer.hyperparameters.loss == "AdaptiveWingLoss" else L2Loss().cuda()

    def train(self, t: float) -> None:
        self.model = self.model.train()

        total_loss = 0.0
        pbar = tqdm(self.train_loader, desc="[LocNet][Trainer][Synthetic] Training")
        for render, mask, heatmap in pbar:
            render, mask, heatmap = render.cuda(), mask.cuda(), heatmap.cuda()
            mask, _ = torch.max(torch.cat([mask.unsqueeze(-1), torch.ones_like(mask).unsqueeze(-1) * t], dim=-1), axis=-1)

            self.optim.zero_grad()

            _heatmap = self.model(render)
            loss = self.loss(_heatmap, heatmap, mask)

            loss.backward()
            self.optim.step()
            
            total_loss += loss.item()
            pbar.set_postfix(loss=total_loss / len(self.train_loader))

    def valid(self) -> None:
        self.model = self.model.eval()

        total_loss = 0.0
        pbar = tqdm(self.valid_loader, desc="[LocNet][Trainer][Synthetic] Validation")
        with torch.no_grad():
            for render, mask, heatmap in pbar:
                render, mask, heatmap = render.cuda(), mask.cuda(), heatmap.cuda()

                _heatmap = self.model(render)
                loss = self.loss(_heatmap, heatmap)
                
                total_loss += loss.item()
                pbar.set_postfix(loss=total_loss / len(self.valid_loader))

    def test(self) -> None:
        self.model = self.model.eval()

        total_loss = 0.0
        pbar = tqdm(self.test_loader, desc="[LocNet][Trainer][Synthetic] Testing")
        with torch.no_grad():
            for render, mask, heatmap in pbar:
                render, mask, heatmap = render.cuda(), mask.cuda(), heatmap.cuda()

                _heatmap = self.model(render)
                loss = self.loss(_heatmap, heatmap)
                
                total_loss += loss.item()
                pbar.set_postfix(loss=total_loss / len(self.test_loader))

    def debug(self, t: float, epoch_pbar: str) -> None:
        self.model = self.model.eval()

        with torch.no_grad():
            render, mask, heatmap = self.test_dataset[0]
            mask, _ = torch.max(torch.cat([mask.unsqueeze(-1), torch.ones_like(mask).unsqueeze(-1) * t], dim=-1), axis=-1)

            _heatmap = self.model(render.unsqueeze(0).cuda()).cpu()
            _heatmap =             _heatmap.permute((1, 2, 0)).numpy()
            render   =               render.permute((1, 2, 0)).numpy()
            mask     =    mask.unsqueeze(0).permute((1, 2, 0)).numpy()
            heatmap  = heatmap.unsqueeze(0).permute((1, 2, 0)).numpy()
        
        img_render  = np.hstack([               render,     mask.repeat(3, -1)])
        img_heatmap = np.hstack([heatmap.repeat(3, -1), _heatmap.repeat(3, -1)])

        img = np.vstack([img_render, img_heatmap]).clip(0, 1)
        img = Image.fromarray((img * 255).astype(np.uint8))
        img.show(title=f"Epoch {epoch_pbar}")

    def __call__(self, epochs: int, debug: bool = False) -> None:
        print("[LocNet][Trainer][Synthetic] Training")
        step_size = epochs / (self.steps + 1)
        spread_diff = self.spread.start - self.spread.end

        for epoch in range(epochs):
            mask = (epoch // step_size) / self.steps
            spread = self.spread.start - spread_diff * (epoch // step_size) / self.steps
            epoch_pbar = f"[{epoch + 1:0{len(str(epochs))}d}/{epochs}]"
            print(f"[LocNet][Trainer][Synthetic] Training Epoch {epoch_pbar}, mask={1 - mask:.2f}, spread={spread:.2f}")
            self.train_loader.dataset.spread = spread
            self.valid_loader.dataset.spread = spread
            self.train(mask)
            self.valid()
            if debug:
                self.debug(mask, epoch_pbar)
        self.test()