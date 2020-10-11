import bpy
import yaml

from typing import List
from typing import Tuple


class PoolData:
    def __init__(self) -> None:
        self.balls: List[Tuple[List[int], int, str]] = []
        self.cues: List[Tuple[List[int], List[int], List[int]]] = []
        self.table: List[List[int]] = []

    def __len__(self) -> int:
        return len(self.balls) + len(self.cues)

    def reset(self) -> None:
        self.balls = []
        self.cues = []

    def append_ball(self, position: List[int], radius: int, color: str) -> None:
        self.balls.append((position, radius, color))

    def append_cue(self, position: List[int], target: List[int], radius: int) -> None:
        self.cues.append((position, target, radius))

    def append_table(self, a: List[int], b: List[int], c: List[int], d: List[int]) -> None:
        self.table = [a, b, c, d]

    def save(self, path: str) -> None:
        balls = [{ "position": p, "radius": r, "color": c } for p, r, c in self.balls]
        balls = balls if len(balls) else None

        cues = [{ "position": p, "target": t, "radius": r } for p, t, r in self.cues]
        cues = cues if len(cues) else None

        with open(path, "w") as f:
            yaml.dump({ "balls": balls, "cues": cues, "table": self.table }, f)

    def __repr__(self) -> str:
        class_name = self.__class__.__name__
        return f"{class_name}(balls={len(self.balls)}, cues={len(self.cues)})"