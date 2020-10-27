from PIL import Image, ImageDraw
from scipy.stats import multivariate_normal
from typing import List, Tuple

import numpy as np


DataFileContent = Tuple[
    List[Tuple[int, ...]],
    List[Tuple[float, ...]],
    List[Tuple[int, ...]]
]
Size = Tuple[int, int]
Point = Tuple[int, int]
Points = List[Point]


def parse_data_file(content: str) -> DataFileContent:
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

    return balls, cues, mask


def create_mask(size: Size, *, corners: Points) -> np.ndarray:
    mask = Image.new(mode='L', size=size)
    draw = ImageDraw.Draw(mask)
    draw.polygon(tuple(corners), fill="white", outline=None)
    return np.array(mask)


def create_gaussian(size: Size, *, point: Point, spread: float) -> np.ndarray:
    w, h = size
    grid = np.dstack(np.mgrid[0:h, 0:w])
    gauss = multivariate_normal.pdf(grid, mean=point, cov=spread)
    return gauss


def create_heatmap(size: Size, *, points: Points, spread: float) -> np.ndarray:
    heatmap = np.concatenate([
        create_gaussian(size, point=p, spread=spread)[:, :, None]
        for p in points
    ], axis=-1).mean(axis=-1)
    return heatmap / heatmap.max()