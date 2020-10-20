import os
import numpy as np
import torch
import yaml

from PIL import Image
from snook.data.dataset.utils import RandomGaussianBlur
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm
from typing import List
from typing import Tuple


class DirDataset(Dataset):
    def __init__(self, render: str, data: str, size: int = 256, window: int = 32, train: bool = False) -> None:
        self.render = sorted([os.path.join(render, f) for f in os.listdir(render)])
        self.data = sorted([os.path.join(data, f) for f in os.listdir(data)])
        self.size = size
        self.window = window
        self.offset = self.window // 2

        if train:
            self.transforms = transforms.Compose([
                transforms.ToPILImage(),
                transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
                RandomGaussianBlur(3),
                transforms.ToTensor(),
            ])
        else:
            self.transforms = transforms.Compose([transforms.ToPILImage(), transforms.ToTensor()])

        lengths = [len(self._valid_cues(i)) for i in tqdm(range(len(self.render)))]
        self.cumulengths = np.cumsum([0] + lengths)

    def __len__(self) -> int:
        return self.cumulengths[-1]
    
    def _valid_cues(self, idx: int) -> List[Tuple[np.ndarray, int]]:
        with open(self.data[idx], "r") as f:
            data = yaml.load(f, Loader=yaml.FullLoader)
        
        cues = []
        if data["cues"] is not None:
            pos_tar = [(np.array(cue["position"]), np.array(cue["target"])) for cue in data["cues"]]
            pos_tar = [e for e in pos_tar if np.all(np.all(e[0] > 0 + self.offset) and np.all(e[0] < self.size - self.offset))]
            if len(pos_tar) > 0:
                cues = pos_tar
        return cues

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        over_idx = (self.cumulengths <= idx).sum() - 1
        offset = self.cumulengths[over_idx]

        objects = self._valid_cues(over_idx)
        pos, tar = objects[idx - offset]

        dir = tar - pos
        dir = dir / (np.sqrt(np.sum(dir * dir)) + 1e-11)
        dir = torch.tensor(dir, dtype=torch.float32)

        render = np.array(Image.open(self.render[over_idx]).convert("RGB"))
        render = render[pos[1] - self.offset:pos[1] + self.offset, pos[0] - self.offset:pos[0] + self.offset]
        render = self.transforms(render)
        
        return render, dir