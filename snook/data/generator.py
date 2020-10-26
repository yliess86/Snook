from abc import ABC, abstractmethod
from mathutils import Euler, Vector
from typing import Any, Callable, Iterator, List, NamedTuple, Sequence, Tuple

import math
import numpy as np
import os
import snook.data.blender as blender


COLORS = ["black", "white", "yellow", "red"]


iRange = Tuple[int, int]
fRange = Tuple[float, float]
cSampling = Callable[..., Vector]


def instantiate_fbxs(path: str, n: int) -> List[blender.Fbx]:
    return [blender.Fbx(path, id=i) for i in range(n)]


def sample_plane_vectors(
    n: int, *, distance: float, sample: cSampling
) -> List[Vector]:
    positions: List[Vector] = []
    for i in range(n):
        while len(positions) < i:
            pos = sample()
            for other in positions:
                dist = np.abs(np.array(other - pos))
                if np.all(dist < distance): continue
            positions.append(pos)
    return positions


def sample_ehmisphere_vector(pos: Vector, radius: fRange) -> Vector:
    r = np.random.uniform(*radius)
    theta = 2 * np.pi * np.random.uniform(0, 2 * np.pi)
    phi = np.arccos(2 * np.random.rand() - 1)
    return pos + Vector((
        r * np.sin(phi) * np.cos(theta),
        r * np.sin(phi) * np.sin(theta),
        np.abs(r * np.cos(phi)),
    )) 


class Sampler(ABC):
    @abstractmethod
    def sample(self) -> Any:
        raise NotImplementedError


class Samplers:
    def __init__(self, samplers: List[Sampler] = []) -> None:
        self.samplers = samplers

    def __len__(self) -> int:
        return len(self.samplers)

    def __getitem__(self, idx: int) -> Sampler:
        return self.samplers[idx]

    def __setitem__(self, idx: int, sampler: Sampler) -> None:
        self.samplers[idx] = sampler

    def __add__(self, sampler: Sampler) -> "Samplers":
        self.samplers.append(sampler)
        return self

    def sample(self) -> None:
        for sampler in self.samplers:
            sampler.sample()


class Plane(Sampler):
    def __init__(self, plane: fRange, height: float) -> None:
        self.plane = plane
        self.height = height

    def __len__(self) -> int:
        return len(self.plane)

    def __getitem__(self, idx: int) -> float:
        return self.plane[idx]

    def sample(self) -> Vector:
        return Vector((
            np.random.uniform(self.plane[0], - self.plane[0]),
            np.random.uniform(self.plane[1], - self.plane[1]),
            self.height,
        ))


class Hdris(Sampler):
    def __init__(self, root: str, *, rot: fRange, scale: fRange) -> None:
        files = filter(lambda f: f.endswith(".hdr"), os.listdir(root))
        files = map(lambda f: os.path.join(root, f), files)
        self.files = sorted(list(files))
        assert len(self), f"{root} does not seem to contain any `.hdr` file"

        self.hdri = blender.HDRI(self.files[0])
        self.rot = math.radians(rot[0]), math.radians(rot[1])
        self.scale = scale

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> str:
        return self.files[idx]

    def sample(self) -> None:
        self.hdri.img = self.files[np.random.randint(0, len(self))]
        self.hdri.rot = Euler(tuple(np.random.uniform(*self.rot, size=3)))
        self.hdri.scale = Vector((np.random.uniform(*self.scale), ) * 3)


class Balls(Sampler):
    def __init__(
        self,
        fbxs: List[str],
        *,
        plane: Plane,
        diameter: float,
        p: float,
        n: int,
    ) -> None:
        r = 0.5 * diameter
        self.fbxs = fbxs
        self.plane = Plane((plane[0] - r, plane[1] - r), plane.height)
        self.diameter = diameter
        self.p = p
        self.n = n
        
        self.balls: Sequence[blender.Fbx] = []
        for fbx in fbxs:
            self.balls += instantiate_fbxs(fbx, n)
        
    def __len__(self) -> int:
        return len(self.balls)

    def __getitem__(self, idx: int) -> blender.Fbx:
        return self.balls[idx]

    def __iter__(self) -> Iterator[blender.Fbx]:
        return iter(self.balls)

    def sample(self) -> None:
        positions = sample_plane_vectors(
            len(self), distance=self.diameter, sample=self.plane.sample
        )
        for ball, pos in zip(self.balls, positions):
            ball.visible = np.random.rand() < self.p
            ball.pos = pos + Vector((0.0, 0.0, 0.5 * self.diameter))
            ball.rot = Euler(tuple(np.random.uniform(
                math.radians(0), math.radians(360), size=3
            )))


class Cues(Sampler):
    def __init__(
        self,
        fbx: str,
        *,
        balls: Balls,
        plane: Plane,
        distance: float,
        p: float,
        n: int,
    ) -> None:
        self.fbx = fbx
        self.balls = balls
        self.plane = plane
        self.distance = distance
        self.range = (0.5 * balls.diameter, self.distance)
        self.p = p
        self.n = n

        self.cues = instantiate_fbxs(fbx, n)

    def __len__(self) -> int:
        return len(self.cues)

    def __getitem__(self, idx: int) -> blender.Fbx:
        return self.cues[idx]

    def __iter__(self) -> Iterator[blender.Fbx]:
        return iter(self.cues)

    def sample(self) -> None:
        sample = lambda: \
            self.plane.sample() if np.random.rand() < 0.5 \
            else self.balls[int(np.random.uniform(0, len(self.balls)))].pos
            
        positions = sample_plane_vectors(
            len(self), distance=1e-2, sample=self.plane.sample
        )
        for cue, target in zip(self.cues, positions):
            cue.visible = np.random.rand() < self.p
            cue.pos = sample_ehmisphere_vector(target, self.range)
            cue.look_at(target, "-Z", "Y")


class Pool(Sampler):
    def __init__(self, fbx: str, *, balls: Balls, cues: Cues) -> None:
        self.fbx = fbx
        self.balls = balls
        self.cues = cues
        self.pool = blender.Fbx(fbx)

    def sample(self) -> None:
        self.balls.sample()
        self.cues.sample()


class Camera(Sampler):
    def __init__(
        self,
        camera: blender.Camera,
        *,
        plane: Plane,
        distance: fRange,
        eps: float = 1e-2,
    ) -> None:
        self.camera = camera
        self.plane = plane
        self.distance = distance
        self.eps = eps

    def sample(self) -> None:
        eps_target = Vector(tuple(np.random.random(size=3) * self.eps))
        eps_pos = Vector(tuple(np.random.random(size=3) * self.eps))
        
        target = Vector((0, 0, self.plane.height)) + eps_target
        pos = Vector((0, 0, self.plane.height)) + eps_pos
        pos.z += np.random.uniform(*self.distance)

        self.camera.pos = pos
        self.camera.look_at(target, "-Z", "Y")


class cFiles(NamedTuple):
    balls: List[str]
    cue: str
    pool: str
    hdris: str


class cTable(NamedTuple):
    plane: fRange
    border: fRange
    height: float


class cDistances(NamedTuple):
    ball_d: float
    cue: float
    camera: fRange


class cProbas(NamedTuple):
    ball: float = 0.5
    cue: float = 0.5
    n: int = 6


class cSensor(NamedTuple):
    fit: str
    size: iRange
    lens: float


class cRender(NamedTuple):
    size: iRange
    tile: iRange
    samples: int
    denoise: bool
    cuda: bool


class Scene:
    def __init__(
        self,
        files: cFiles,
        table: cTable,
        distances: cDistances,
        probas: cProbas = cProbas(0.5, 0.5, 6),
        sensor: cSensor = cSensor("HORIZONTAL", (36, 24), 85.0),
        render: cRender = cRender((512, 512), (64, 64), 64, False, True),
    ) -> None:
        plane = Plane((
            table.plane[0] - 0.5 * distances.ball_d,
            table.plane[1] - 0.5 * distances.ball_d
        ), table.height)

        self.balls = Balls(
            files.balls,
            plane=plane,
            diameter=distances.ball_d,
            p=probas.ball,
            n=probas.n,
        )
        
        self.cues = Cues(
            files.cue,
            balls=self.balls,
            plane=plane,
            distance=distances.cue,
            p=probas.cue,
            n=probas.n,
        )
        
        self.camera = blender.Camera(
            "Camera",
            sensor_fit=sensor.fit,
            sensor_size=sensor.size,
            lens=sensor.lens,
        )

        self.samplers = Samplers([
            Pool(files.pool, balls=self.balls, cues=self.cues),
            Hdris(files.hdris, rot=(0, 360), scale=(0.5, 25.0)),
            Camera(self.camera, plane=plane, distance=distances.camera)
        ])
        
        self.cycle = blender.Cycle(
            render.size,
            self.camera,
            tile=render.tile,
            samples=render.samples,
            denoise=render.denoise,
            cuda=render.cuda,
        )

        self.mask = (
            table.plane[0] + table.border[0], table.plane[1] + table.border[1],
        )

    def sample(self) -> None:
        self.samplers.sample()

    def render(self, path: str) -> None:
        self.cycle.render(path)

    def register(self, path: str) -> None:
        mask: List[Tuple[int, int]] = []
        for i, j in [(-1, -1), (1, -1), (1, 1), (-1, 1)]:
            corner = Vector((self.mask[0] * i, self.mask[1] * j, 0.0))
            coords = self.camera.ndc(corner)
            mask.append((int(coords.x), int(coords.y)))

        def in_view(pos: Vector) -> bool:
            xy = np.abs(np.array(pos.xy))
            resolution = np.array(self.cycle.resolution)
            return np.all(xy > 0) and np.all(xy < resolution)

        balls: List[Tuple[int, int, int]] = []
        for ball in self.balls:
            ball_coords = self.camera.ndc(ball.pos)
            if not in_view(ball_coords): continue
            for color in COLORS:
                if color in ball.obj.name:
                    ball_color = COLORS.index(color)
                    break
            balls.append((int(ball_coords.x), int(ball_coords.y), ball_color))

        cues: List[Tuple[int, int, float, float]] = []
        for cue in self.cues:
            cue_coords = self.camera.ndc(cue.pos)
            if not in_view(cue_coords): continue
            cue_dir = cue.rot.to_quaternion() @ Vector((0.0, 0.0, -1.0))
            cue_dir_a = self.camera.ndc(cue.pos)
            cue_dir_b = self.camera.ndc(cue.pos + cue_dir * 10)
            cue_dir = (cue_dir_b - cue_dir_a).normalized()
            cues.append((
                int(cue_coords.x), int(cue_coords.y), cue_dir.x, cue_dir.y,
            ))

        with open(path, "w") as f:
            f.write(f"[{len(balls)} balls] ndc_x ndc_y label\n")
            for x, y, c in balls:
                f.write(f"{x} {y} {c}\n")
            f.write("\n")

            f.write(f"[{len(cues)} cues] ndc_x ndc_y dir_x dir_y\n")
            for x, y, dx, dy in cues:
                f.write(f"{x} {y} {dx} {dy}\n")
            f.write("\n")

            f.write(f"[{len(mask)} mask] ndc_x ndc_y\n")
            for x, y in mask:
                f.write(f"{x} {y}\n")