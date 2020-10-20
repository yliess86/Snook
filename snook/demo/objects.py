import cv2
import numpy as np
import torch

from skimage.transform import resize
from typing import List
from typing import NamedTuple


class Vec2(NamedTuple):
    x: float
    y: float


class Cue:
    COLOR = (0.1, 0.2, 0.8, 0.5)

    def __init__(self, position: np.ndarray, direction: np.ndarray) -> None:
        self.position = Vec2(*position[::-1])
        self.target = Vec2(*(direction[::-1] * 1000))

    def draw(self, img: np.ndarray, scale: np.ndarray) -> None:
        position = int(self.position.x * scale[0]), int(self.position.y * scale[1])
        target   = int(self.target.x   * scale[0]), int(self.target.y   * scale[1])
        cv2.circle(img, position, radius=18, color=self.COLOR, thickness=4)
        cv2.line(img, position, target, color=self.COLOR, thickness=4)


class Ball:
    BLACK       = (0.2, 0.2, 0.2, 0.5)
    WHITE       = (0.8, 0.8, 0.8, 0.5)
    RED         = (0.8, 0.1, 0.2, 0.5)
    YELLOW      = (0.6, 0.6, 0.2, 0.5)
    COLORS      = [BLACK, WHITE, YELLOW, RED]
    LABEL2COLOR = { label: color for label, color in enumerate(COLORS) }

    def __init__(self, position: np.ndarray, label: int) -> None:
        self.position = Vec2(*position[::-1])
        self.label = label

    def draw(self, img: np.ndarray, scale: np.ndarray) -> None:
        position = int(self.position.x * scale[0]), int(self.position.y * scale[1])
        cv2.circle(img, position, radius=18, color=self.LABEL2COLOR[self.label], thickness=4)


class Debug:
    def __init__(self, capture: np.ndarray, mask: np.ndarray, heatmap: np.ndarray) -> None:
        self.capture = capture
        self.mask = mask
        self.heatmap = heatmap

    def draw(self, img: np.ndarray, size: int, scale: np.ndarray) -> None:
        dsize = (1 / scale * size).astype(int)
        img[:3 * dsize[0], :dsize[1], :] = np.vstack([
            resize(self.capture, dsize),
            resize(self.mask, dsize),
            resize(self.heatmap, dsize),
        ])


class Pool:
    def __init__(self, balls: List[Ball] = [], cues: List[Cue] = [], debug: Debug = None) -> None:
        self.balls = balls
        self.cues = cues
        self.debug = debug

    def draw(self, img: np.ndarray, scale: float, debug: bool = True) -> None:
        for ball in self.balls:
            ball.draw(img, scale)
        
        for cue in self.cues:
            cue.draw(img, scale)

        if debug and self.debug is not None:
            self.debug.draw(img, 256, scale)