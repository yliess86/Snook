import os
import numpy as np
import torch
import yaml

from PIL import Image
from PIL import ImageDraw
from torch.utils.data import Dataset
from torchvision import transforms
from typing import Dict
from typing import Tuple


class MaskDataset(Dataset):
    def __init__(self, render: str, data: str, train: bool = False) -> None:
        self.render = sorted([os.path.join(render, f) for f in os.listdir(render)])
        self.data = sorted([os.path.join(data, f) for f in os.listdir(data)])
        if train:
            self.transforms = transforms.Compose([
                transforms.ColorJitter(0.2, 0.2, 0.2, 0.2),
                transforms.ToTensor(),
                transforms.RandomErasing(),
            ])
        else:
            self.transforms = transforms.Compose([transforms.ToTensor()])

    def __len__(self) -> int:
        return len(self.render)

    def _build_mask(self, data: Dict, img: Image) -> np.ndarray:
        mask = Image.new(mode='L', size=(img.width, img.height))
        draw = ImageDraw.Draw(mask)
        draw.polygon(tuple(tuple(corner) for corner in data["table"]), fill="white", outline=None)
        
        return np.array(mask)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img = Image.open(self.render[idx]).convert("RGB")
        with open(self.data[idx], "r") as f:
            data = yaml.load(f, Loader=yaml.FullLoader)

        render = self.transforms(img)
        mask   = torch.tensor(self._build_mask(data, img) / 255.0, dtype=torch.float32)

        return render, mask