import os
import numpy as np
import torch
import yaml

from PIL import Image
from PIL import ImageDraw
from scipy.stats import multivariate_normal
from torch.utils.data import Dataset
from torchvision import transforms
from typing import Dict
from typing import Tuple


class LocDataset(Dataset):
    def __init__(self, render: str, data: str, spread: float = 10.0, train: bool = False) -> None:
        self.render = sorted([os.path.join(render, f) for f in os.listdir(render)])
        self.data = sorted([os.path.join(data, f) for f in os.listdir(data)])
        self.spread = spread

        if train:
            self.transforms = transforms.Compose([
                transforms.ColorJitter(0.2, 0.2, 0.2, 0.2),
                transforms.ToTensor(),
            ])
        else:
            self.transforms = transforms.Compose([transforms.ToTensor()])

    def __len__(self) -> int:
        return len(self.render)

    def _build_heatmaps(self, data: Dict, img: Image) -> Tuple[np.ndarray, np.ndarray]:
        mask = Image.new(mode='L', size=(img.width, img.height))
        draw = ImageDraw.Draw(mask)
        draw.polygon(tuple(tuple(corner) for corner in data["table"]), fill="white", outline=None)
        mask = np.array(mask)

        balls = np.zeros((img.height, img.width, 6 * 2 + 2))
        if "balls" in data and data["balls"] is not None:
            for b, ball in enumerate(data["balls"]):
                grid = np.dstack(np.mgrid[0:img.width, 0:img.height])
                gauss = multivariate_normal.pdf(grid, mean=tuple(ball["position"]), cov=ball["radius"] * self.spread).T
                balls[..., b] = gauss / np.max(gauss)
        balls = np.max(balls, axis=-1)

        cues = np.zeros((img.height, img.width, 2))
        if "cues" in data and data["cues"] is not None:
            for c, cue in enumerate(data["cues"]):
                grid = np.dstack(np.mgrid[0:img.width, 0:img.height])
                gauss = multivariate_normal.pdf(grid, mean=tuple(cue["position"]), cov=cue["radius"] * self.spread).T
                cues[..., c] = gauss / np.max(gauss)
        cues = np.max(cues, axis=-1)

        heatmaps = np.zeros((img.height, img.width, 2))
        heatmaps[:, :, 0] = balls
        heatmaps[:, :, 1] = cues
        heatmap = np.max(heatmaps, axis=-1)

        return mask, heatmap

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        img = Image.open(self.render[idx]).convert("RGB")
        with open(self.data[idx], "r") as f:
            data = yaml.load(f, Loader=yaml.FullLoader)

        render = self.transforms(img)
        mask, heatmap = self._build_heatmaps(data, img)
        
        mask    = torch.tensor(mask / 255.0, dtype=torch.float32)
        heatmap = torch.tensor(heatmap,      dtype=torch.float32)

        return render, mask, heatmap