import os
import numpy as np
import torch
import yaml

from PIL import Image
from PIL import ImageDraw
from scipy.stats import multivariate_normal
from snook.data.dataset.utils import RandomGaussianBlur
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
                RandomGaussianBlur(3),
                transforms.ToTensor(),
            ])
        else:
            self.transforms = transforms.Compose([transforms.ToTensor()])

    def __len__(self) -> int:
        return len(self.render)

    def _build_items(self, data: Dict, width: int, height: int) -> Tuple[np.ndarray, np.ndarray]:
        mask = Image.new(mode='L', size=(width, height))
        draw = ImageDraw.Draw(mask)
        draw.polygon(tuple(tuple(corner) for corner in data["table"]), fill="white", outline=None)
        mask = np.array(mask) / 255.0

        heatmaps = []
        if "balls" in data and data["balls"] is not None:
            for ball in data["balls"]:
                grid = np.dstack(np.mgrid[0:width, 0:height])
                gauss = multivariate_normal.pdf(grid, mean=ball["position"], cov=self.spread).T
                heatmaps.append(gauss[..., None])
        
        if "cues" in data and data["cues"] is not None:
            for cue in data["cues"]:
                grid = np.dstack(np.mgrid[0:width, 0:height])
                gauss = multivariate_normal.pdf(grid, mean=cue["position"], cov=self.spread).T
                heatmaps.append(gauss[..., None])

        heatmap = np.concatenate(heatmaps, axis=-1) if len(heatmaps) > 0 else np.zeros((height, width, 1), dtype=np.float32)
        heatmap = heatmap.mean(axis=-1)
        heatmap = heatmap / (heatmap.max() if heatmap.max() > 0 else 1.0)

        return mask, heatmap

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        img = Image.open(self.render[idx]).convert("RGB")
        with open(self.data[idx], "r") as f:
            data = yaml.load(f, Loader=yaml.FullLoader)

        render = self.transforms(img)
        mask, heatmap = self._build_items(data, img.width, img.height)
        
        mask    = torch.tensor(mask,    dtype=torch.float32)
        heatmap = torch.tensor(heatmap, dtype=torch.float32)

        return render, mask, heatmap


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    spreads = [1000, 500, 250, 125, 75, 35, 10, 8]
    dataset = LocDataset("resources/data/dataset/test/render", "resources/data/dataset/test/data", 0, False)
    for i, s in enumerate(spreads):
        dataset.spread = s
        plt.subplot(1, len(spreads), i + 1)
        plt.imshow(dataset[0][-1].detach().cpu().numpy())
        plt.axis("off")
    plt.tight_layout()
    plt.show()