from PIL import Image, ImageDraw
from scipy.ndimage import rotate
from scipy.signal import convolve2d
from scipy.stats import multivariate_normal
from snook.data.generator import COLORS
from torch.utils.data import Dataset
from torchvision.transforms import Compose, ToTensor
from typing import Any, Callable, List, NamedTuple, Sequence, Tuple

import os
import numpy as np
import torch
import torch.nn.functional as F


DataFileContent = Tuple[
    List[Tuple[int, int, int]],
    List[Tuple[int, int, float, float]],
    List[Tuple[int, int]]
]
Size = Tuple[int, int]
Point = Tuple[int, int]
Points = List[Point]
Range = Tuple[float, float]
ReMaHe = Tuple[torch.Tensor, torch.Tensor,torch.Tensor]
TemporalReHe = Tuple[torch.Tensor, torch.Tensor]
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


def create_linear_motionblur_kernel(length: int, angle: float) -> torch.Tensor:
    kernel = np.zeros((length, length))
    center = length // 2
    kernel[center, :] = 1.0
    kernel = rotate(kernel, angle, reshape=False)
    return torch.Tensor(kernel / np.sum(kernel))


class RandomLinearMotionBlur:
    def __init__(self, length: Range, angle: Range, p: float = 0.5) -> None:
        self.length = length
        self.angle = angle
        self.p = p

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        if np.random.rand() > self.p:
            return img
        
        length = int(np.random.uniform(*self.length))
        angle = int(np.random.uniform(*self.angle))
        kernel = create_linear_motionblur_kernel(length, angle)
        kernel = kernel.unsqueeze(0).repeat(img.size(0), 1, 1)

        img = img.unsqueeze(0)        # ( B, iC,     iH, iW)
        kernel = kernel.unsqueeze(1)  # (oC, iC / G, kH, kW)

        pad = ((length - 1) // 2)
        even = int((length % 2) == 0)
        padding = [pad + even, pad] * 2 + [0] * 4
        
        img = F.pad(img, padding)
        img = F.conv2d(img, kernel, groups=img.size(1))
        return img.squeeze(0)


class ResizeSquare:
    def __init__(self, size: int) -> None:
        self.size = size

    def __call__(self, img: Image) -> Image:
        ratio = max(img.width / self.size, img.height / self.size)
        img = img.resize((int(img.width / ratio), int(img.height / ratio)))
        img = np.array(img) / 255.0

        h, w, c = img.shape
        offset_w, offset_h = (self.size - w) // 2, (self.size - h) // 2
        
        new_img = np.zeros((self.size, self.size, c))
        new_img[offset_h:offset_h + h, offset_w:offset_w + w, :] = img
        
        return Image.fromarray((new_img * 255.0).astype(np.uint8))

    def reposition(self, points: Points, size: Size) -> Points:
        w, h = size
        ratio = max(w / self.size, h / self.size)
        points = np.array(points) / ratio
        
        w, h = int(w / ratio), int(h / ratio)
        points[:, 0] += (self.size - w) // 2
        points[:, 1] += (self.size - h) // 2

        return points.tolist()


class ReMaHeDataset(Dataset):
    def __init__(
        self,
        renders: str,
        data: str,
        *,
        spread: float = 4.0,
        train: bool = False,
        transforms: List[Callable[..., Any]] = [ToTensor()],
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
        self.transforms = Compose([*transforms])

    def __len__(self) -> int:
        return len(self.renders)

    def __getitem__(self, idx: int) -> ReMaHe:
        render = Image.open(self.renders[idx]).convert("RGB")
        with open(self.data[idx], "r") as f:
            balls, cues, mask = parse_data_file(f.read())
        landmarks = [ball[:2] for ball in balls] + [cue[:2] for cue in cues]

        size = render.width, render.height
        mask = create_mask(size, corners=mask)
        heatmap = create_heatmap(size, points=landmarks, spread=self.spread)

        return (
            self.transforms(render),
            torch.from_numpy(mask),
            torch.from_numpy(heatmap),
        )


class TemporalReHeDataset(Dataset):
    def __init__(
        self,
        renders: str,
        data: str,
        *,
        spread: float = 4.0,
        train: bool = False,
        transforms: List[Callable[..., Any]] = [ToTensor()],
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
        self.transforms = Compose([*transforms])

    def __len__(self) -> int:
        return len(self.renders) - 1

    def __getitem__(self, idx: int) -> TemporalReHe:
        transforms = self.transforms.transforms
        resize_trans = [t for t in transforms if type(t) == ResizeSquare]
        reposition = lambda x, size: (
            resize_trans[0].reposition(x, size) if len(resize_trans) else x
        )
        
        def generate_data(i: int) -> Tuple[torch.Tensor, torch.Tensor]:
            render = Image.open(self.renders[i]).convert("RGB")
            size = render.width, render.height
            with open(self.data[idx], "r") as f:
                balls, cues, _ = parse_data_file(f.read())
            points = [ball[:2] for ball in balls] + [cue[:2] for cue in cues]
            points = reposition(points, size)
            size = [resize_trans[0].size] * 2 if len(resize_trans) else size
            heatmap = create_heatmap(size, points=points, spread=self.spread)
            return render, heatmap

        render_0, heatmap_0 = generate_data(idx)
        render_1, heatmap_1 = generate_data(idx + 1)

        return (
            torch.cat([
                self.transforms(render_0).unsqueeze(0),
                self.transforms(render_1).unsqueeze(0),
            ]),
            torch.cat([
                torch.from_numpy(heatmap_0).unsqueeze(0),
                torch.from_numpy(heatmap_1).unsqueeze(0),
            ]),
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
        transforms: List[Callable[..., Any]] = [ToTensor()],
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
        self.transforms = Compose([*transforms])
        
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
        
        render = np.array(Image.open(path).convert("RGB")) / 255.0
        
        offset = self.window // 2
        view = render[y - offset:y + offset, x - offset:x + offset, :]
        
        window = np.zeros((self.window, self.window, 3))
        window[:view.shape[0], :view.shape[1], :] = view

        window = Image.fromarray((window * 255.0).astype(np.uint8))
        window = self.transforms(window)

        return window, label