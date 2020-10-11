import numpy as np
import os

from typing import Any
from typing import List
from typing import Optional
from typing import Tuple


Range = Tuple[float, float]


class Sampler:
    def samle(self) -> Any:
        raise NotImplementedError("[Snook][Generator][Sampler] sample() needs to overritten to be called")


class PoolSampler(Sampler):
    def __init__(self, half_width: float, half_height: float, depth: float, safe_dist: Optional[float] = None) -> None:
        super(PoolSampler, self).__init__()
        self.half_width = half_width
        self.half_height = half_height
        self.depth = depth
        self.safe_dist = safe_dist
        self.positions: List[np.ndarray] = []

    def __len__(self) -> int:
        return len(self.positions)

    def reset(self) -> None:
        self.positions = []

    def _pos_valid(self, pos: np.ndarray) -> bool:
        if self.safe_dist is None or len(self) < 1:
            self.positions.append(pos)
            return True

        positions = np.array(self.positions)
        candidates = np.repeat(np.expand_dims(pos, 0), len(self), axis=0)
        distances = np.sqrt(np.sum((positions - candidates) ** 2, axis=1))
        unsafe_positions = distances <= self.safe_dist
        position_safe = not np.any(unsafe_positions)
        if position_safe:
            self.positions.append(pos)
        return position_safe

    def sample(self) -> np.ndarray:
        pos = None
        while pos is None or not self._pos_valid(pos):
            x = np.random.uniform(-self.half_width, self.half_width)
            y = np.random.uniform(-self.half_height, self.half_height)
            pos = np.array([x, y, self.depth])
        return pos

    def __repr__(self) -> str:
        return (
            f"PlaneSampler("
                f"half_width={self.half_width}, "
                f"half_height={self.half_height}, "
                f"depth={self.depth}, "
                f"safe={self.safe_dist}, "
                f"positions={len(self)}"
            f")"
        )


class HalfOnionSkinSampler(Sampler):
    def __init__(self, pos: np.ndarray, radius_range: Range) -> None:
        super(HalfOnionSkinSampler, self).__init__()
        self.pos = pos
        self.radius_range = radius_range

    def sample(self) -> np.ndarray:
        radius = np.random.uniform(*self.radius_range)
        theta = 2 * np.pi * np.random.uniform(0, 2 * np.pi)
        phi = np.arccos(2 * np.random.random() - 1)
        x = radius * np.sin(phi) * np.cos(theta)
        y = radius * np.sin(phi) * np.sin(theta)
        z = np.abs(radius * np.cos(phi))
        return self.pos + np.array(([x, y, z]))

    def __repr__(self) -> str:
        return (
            f"HalfOnionSkinSampler("
                f"pos={tuple(self.pos)}, "
                f"r1={self.radius_range[0]}, "
                f"r2={self.radius_range[1]}"
            f")"
        )


class HDRISampler(Sampler):
    def __init__(self, root: str) -> None:
        self.root = root
        self.files = [f for f in os.listdir(root) if f.endswith(".hdr")]

    def __len__(self) -> int:
        return len(self.files)

    def sample(self) -> str:
        f = self.files[np.random.choice(len(self))]
        return os.path.join(self.root, f)

    def __repr__(self) -> str:
        return f"HDRISampler(root={self.root}, size={len(self)})"