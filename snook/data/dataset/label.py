import os
import numpy as np
import torch
import yaml

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm
from typing import List
from typing import Tuple


class LabelDataset(Dataset):
    NAME2LABEL = { color: label for label, color in enumerate(["black", "white", "yellow", "red", "cue"]) }
    
    def __init__(self, render: str, data: str, size: int = 256, window: int = 32, train: bool = False) -> None:
        self.render = sorted([os.path.join(render, f) for f in os.listdir(render)])
        self.data = sorted([os.path.join(data, f) for f in os.listdir(data)])
        self.size = size
        self.window = window
        self.offset = window // 2

        if train:
            self.transforms = transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(360),
                transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
                transforms.ToTensor(),
            ])
        else:
            self.transforms = transforms.Compose([transforms.ToPILImage(), transforms.ToTensor()])

        pbar = tqdm(range(len(self.render)))
        lengths = [len(self._valid_balls(i) + self._valid_cues(i)) for i in pbar]
        self.cumulengths = np.cumsum([0] + lengths)

    def __len__(self) -> int:
        return self.cumulengths[-1]

    def _valid_balls(self, idx: int) -> List[Tuple[np.ndarray, int]]:
        with open(self.data[idx], "r") as f:
            data = yaml.load(f, Loader=yaml.FullLoader)

        balls = []
        if data["balls"] is not None:
            pos_label = [(np.array(ball["position"]), self.NAME2LABEL[str(ball["color"])]) for ball in data["balls"]]
            pos_label = [e for e in pos_label if np.all(np.all(e[0] > 0 + self.offset) and np.all(e[0] < self.size - self.offset))]
            if len(pos_label) > 0:
                balls = pos_label
        return balls
    
    def _valid_cues(self, idx: int) -> List[Tuple[np.ndarray, int]]:
        with open(self.data[idx], "r") as f:
            data = yaml.load(f, Loader=yaml.FullLoader)
        
        cues = []
        if data["cues"] is not None:
            pos_label = [(np.array(cue["position"]), self.NAME2LABEL["cue"]) for cue in data["cues"]]
            pos_label = [e for e in pos_label if np.all(np.all(e[0] > 0 + self.offset) and np.all(e[0] < self.size - self.offset))]
            if len(pos_label) > 0:
                cues = pos_label
        return cues

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        over_idx = (self.cumulengths <= idx).sum() - 1
        offset = self.cumulengths[over_idx]

        objects = self._valid_balls(over_idx) + self._valid_cues(over_idx)
        pos, label = objects[idx - offset]

        render = np.array(Image.open(self.render[over_idx]).convert("RGB"))
        render = render[pos[1] - self.offset:pos[1] + self.offset, pos[0] - self.offset:pos[0] + self.offset]
        render = self.transforms(render)

        return render, label