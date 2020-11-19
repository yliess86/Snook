from PIL import Image, ImageDraw
from scipy.stats import multivariate_normal
from snook.data.generator import COLORS
from torch.utils.data import Dataset
from torchvision.transforms import Compose, ToTensor
from typing import List, NamedTuple, Sequence, Tuple

import os
import numpy as np
import torch


DataFileContent = Tuple[
    List[Tuple[int, int, int]],
    List[Tuple[int, int, float, float]],
    List[Tuple[int, int]]
]
Size = Tuple[int, int]
Point = Tuple[int, int]
Points = List[Point]
ReMaHe = Tuple[torch.Tensor, torch.Tensor,torch.Tensor]
Cl = Tuple[torch.Tensor, int]


def parse_data_file(content: str) -> DataFileContent:
    content = content.strip()
    balls_content, cues_content, mask_content, *_ = content.split("\n\n")
    
    balls_data = balls_content.split("\n")[1:]
    cues_data = cues_content.split("\n")[1:]
    mask_data = mask_content.split("\n")[1:]

    balls = [tuple(map(int, data.split(" "))) for data in balls_data]
    cues = [(
        *map(int, data.split(" ")[:2]),
        *map(float, data.split(" ")[2:])
    ) for data in cues_data]
    mask = [tuple(map(int, data.split(" "))) for data in mask_data]

    return balls, cues, mask # type: ignore


def create_mask(size: Size, *, corners: Points) -> np.ndarray:
    mask = Image.new(mode='L', size=size)
    draw = ImageDraw.Draw(mask)
    draw.polygon(tuple(corners), fill="white", outline=None)
    return np.array(mask) / 255.0


def create_gaussian(size: Size, *, point: Point, spread: float) -> np.ndarray:
    w, h = size
    grid = np.dstack(np.mgrid[0:h, 0:w])
    gauss = multivariate_normal.pdf(grid, mean=point, cov=spread)
    return gauss


def create_heatmap(size: Size, *, points: Points, spread: float) -> np.ndarray:
    if len(points) <= 0:
        return np.zeros(size)
        
    heatmap = np.concatenate([
        create_gaussian(size, point=p, spread=spread)[:, :, None]
        for p in points
    ], axis=-1).mean(axis=-1)
    return (heatmap / heatmap.max()).T


class ReMaHeDataset(Dataset):
    def __init__(
        self,
        renders: str,
        data: str,
        *,
        spread: float = 4.0,
        train: bool = False,
    ) -> None:
        self.renders = [
            os.path.join(renders, f) 
            for f in sorted(os.listdir(renders))
            if f.endswith(".png")
        ]
        self.data = [
            os.path.join(data, f) 
            for f in sorted(os.listdir(data))
            if f.endswith(".txt")
        ]
        assert len(self.renders) == len(self.data)

        self.spread = spread
        self.transforms = Compose([ToTensor()])

    def __len__(self) -> int:
        return len(self.renders)

    def __getitem__(self, idx: int) -> ReMaHe:
        render = Image.open(self.renders[idx]).convert("RGB")
        with open(self.data[idx], "r") as f:
            balls, cues, mask = parse_data_file(f.read())
        landmarks = [ball[:2] for ball in balls] + [cue[:2] for cue in cues]

        size = (render.width, render.height)
        mask = create_mask(size, corners=mask)
        heatmap = create_heatmap(size, points=landmarks, spread=self.spread)

        return (
            self.transforms(render),
            torch.from_numpy(mask),
            torch.from_numpy(heatmap),
        )


class ClSample(NamedTuple):
    path: str
    pos: Tuple[int, int]
    label: int


class ClDataset(Dataset):
    def __init__(self,
        renders: str,
        data: str,
        *,
        window: int = 64,
        train: bool = False,
    ) -> None:
        _renders = [
            os.path.join(renders, f) 
            for f in sorted(os.listdir(renders))
            if f.endswith(".png")
        ]
        _data = [
            os.path.join(data, f) 
            for f in sorted(os.listdir(data))
            if f.endswith(".txt")
        ]
        assert len(_renders) == len(_data)

        self.window = window
        self.transforms = Compose([ToTensor()])
        
        self.data: Sequence[ClSample] = []
        for render, datum in zip(_renders, _data):
            with open(datum, "r") as f:
                balls, cues, _ = parse_data_file(f.read())
            self.data += [
                ClSample(render, ball[:2], ball[-1]) for ball in balls
            ]
            self.data += [
                ClSample(render, cue[:2], len(COLORS)) for cue in cues
            ]

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Cl:
        path, (x, y), label = self.data[idx]
        
        render = Image.open(path).convert("RGB")
        render = self.transforms(render)

        offset = self.window // 2
        view = render[:, y - offset:y + offset, x - offset:x + offset]

        window = torch.zeros((3, self.window, self.window), dtype=torch.float)
        window[:, :view.size(1), :view.size(2)] = view

        return window, label